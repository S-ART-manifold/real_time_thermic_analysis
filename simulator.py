import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QRadioButton, 
                             QTabWidget, QSlider, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QPushButton, QComboBox, QDoubleSpinBox, 
                             QCheckBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from stl import mesh as stl_mesh

# Configurar PyQtGraph con paleta oscura aeroespacial
pg.setConfigOption('background', '#161b22')
pg.setConfigOption('foreground', '#c9d1d9')

# Constantes físicas del Aluminio 6061-T6 y del entorno espacial
K_AL = 0.167       # Conductividad térmica: W / (mm * K)
ALPHA_AL = 69.0    # Difusividad térmica: mm^2 / s
R_MIN = 20.0       # Radio mínimo efectivo del nodo (mm)
SIGMA_SB = 5.67037e-8 # Constante de Stefan-Boltzmann: W / (m^2 * K^4)
TOTAL_RAD_AREA = 0.38 # Área superficial radiativa total del 3U + paneles (m^2)
EPSILON_AVG = 0.85    # Emisividad infrarroja media ponderada

# ==================== SHADER DE ILUMINACIÓN SOLAR DESDE +Z ====================
vs_solar = gl.shaders.VertexShader("""
uniform mat4 u_mvp;
uniform mat3 u_normal;
attribute vec4 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;
varying vec4 v_color;
varying vec3 v_normal;
void main() {
    v_normal = normalize(u_normal * a_normal);
    v_color = a_color;
    gl_Position = u_mvp * a_position;
}
""")

fs_solar = gl.shaders.FragmentShader("""
#ifdef GL_ES
precision mediump float;
#endif
varying vec4 v_color;
varying vec3 v_normal;
void main() {
    // Vector de radiación solar directa desde +Z (hacia -Z)
    vec3 sun_dir = normalize(vec3(0.04, 0.04, 1.0));
    float direct_sun = max(0.0, dot(v_normal, sun_dir));
    // Albedo e iluminación difusa de la Tierra (-Z)
    float earth_fill = max(0.0, dot(v_normal, vec3(0.0, 0.0, -1.0))) * 0.16;
    float ambient = 0.28;
    vec3 rgb = v_color.rgb * (ambient + 0.72 * direct_sun + earth_fill);
    gl_FragColor = vec4(rgb, v_color.a);
}
""")

# Registrar el shader en pyqtgraph
prog_solar = gl.shaders.ShaderProgram('solar_shaded', [vs_solar, fs_solar])
if not any(s.name == 'solar_shaded' for s in gl.shaders.Shaders):
    gl.shaders.Shaders.append(prog_solar)

# ==================== UTILIDADES DE COLOR ====================
def create_colormap(min_t=-30, max_t=70, steps=1000):
    """ Colormap térmico orbital calibrado (-30°C a +70°C) """
    temps = np.linspace(min_t, max_t, steps)
    cmap = np.zeros((steps, 4), dtype=np.float32)
    for i, temp in enumerate(temps):
        t_norm = max(0.0, min(1.0, (temp - min_t) / (max_t - min_t)))
        if t_norm < 0.25: 
            r, g, b = 0.0, 4.0 * t_norm, 1.0
        elif t_norm < 0.5: 
            r, g, b = 0.0, 1.0, 1.0 - 4.0 * (t_norm - 0.25)
        elif t_norm < 0.75: 
            r, g, b = 4.0 * (t_norm - 0.5), 1.0, 0.0
        else: 
            r, g, b = 1.0, 1.0 - 4.0 * (t_norm - 0.75), 0.0
        cmap[i] = [r, g, b, 1.0]
    return cmap

class ColorBarWidget(QWidget):
    def __init__(self, min_t=-30, max_t=70):
        super().__init__()
        self.min_t = min_t
        self.max_t = max_t
        self.setFixedHeight(24)
        self.setMinimumWidth(180)

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.00, QColor(0, 0, 255))
        grad.setColorAt(0.25, QColor(0, 255, 255))
        grad.setColorAt(0.50, QColor(0, 255, 0))
        grad.setColorAt(0.75, QColor(255, 255, 0))
        grad.setColorAt(1.00, QColor(255, 0, 0))
        painter.fillRect(0, 2, w, 8, grad)
        painter.setPen(QPen(QColor("#8b949e")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(2, 22, f"{self.min_t}°C")
        painter.drawText(w // 2 - 12, 22, f"{int((self.min_t+self.max_t)/2)}°C")
        painter.drawText(w - 30, 22, f"+{self.max_t}°C")

# ==================== CLASES DE MODELO ====================
class HeatSourceNode:
    def __init__(self, name, subsystem, pos, power, max_power=10.0):
        self.name = name
        self.subsystem = subsystem
        self.pos = np.array(pos, dtype=np.float32)
        self.power = float(power)
        self.max_power = float(max_power)
        self.enabled = True
        self.label_item = None # GLTextItem asociado

class ThermistorSensor:
    def __init__(self, sensor_id, name, subsystem, pos, t_min=-20.0, t_max=60.0):
        self.id = sensor_id
        self.name = name
        self.subsystem = subsystem
        self.pos = np.array(pos, dtype=np.float32)
        self.t_min = t_min
        self.t_max = t_max
        self.temp = 20.0
        self.label_item = None # GLTextItem asociado

class CubeSat3DView(gl.GLViewWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setCameraPosition(distance=950, elevation=25, azimuth=50)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.button() == Qt.LeftButton and self.parent_app.picking_mode:
            self.parent_app.handle_3d_pick(ev.pos())

# ==================== APLICACIÓN PRINCIPAL ====================
class SimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S-ART 3U CubeSat | Simulador Térmico Orbital & Consola de Vuelo")
        self.resize(1650, 920)

        # Variables de tiempo y simulación
        self.sim_time = 0.0
        self.is_paused = False
        self.sim_speed = 50.0
        self.altitude_km = 440.0
        self.eclipse_mode = "AUTO"
        self.in_eclipse = False
        self.picking_mode = False
        self.sun_intensity = 1.0

        # Parámetros globales de radiación al vacío
        self.q_rad_total = 45.0  # Vatios irradiados al vacío
        self.t_rad_equiv = 24.0  # Temperatura de equilibrio radiativo (°C)
        self.q_net = 0.0         # Flujo neto (Qin - Qrad) en Vatios

        # Selección activa para resaltado visual en 3D
        self.selected_source_idx = 0
        self.selected_therm_idx = -1

        # Búfer circular para gráfica temporal
        self.plot_capacity = 250
        self.plot_count = 0
        self.buf_time = np.zeros(self.plot_capacity, dtype=np.float32)
        self.buf_pan = np.zeros(self.plot_capacity, dtype=np.float32)
        self.buf_top = np.zeros(self.plot_capacity, dtype=np.float32)
        self.buf_obc = np.zeros(self.plot_capacity, dtype=np.float32)
        self.buf_bat = np.zeros(self.plot_capacity, dtype=np.float32)
        self.buf_bot = np.zeros(self.plot_capacity, dtype=np.float32)

        # Inicialización de subsistemas y GUI
        self.init_thermal_architecture()
        self.init_gui()
        self.load_full_cad_model()

        # Timers:
        # 1. Física diferencial continua a 20 Hz (50 ms)
        self.phys_timer = QTimer()
        self.phys_timer.timeout.connect(self.update_physics)
        self.phys_timer.start(50)

        # 2. Render 3D térmico a 3.0 Hz (333 ms)
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update_3d_render)
        self.render_timer.start(333)

        # 3. Gráfica temporal y actualización de etiquetas a 5.0 Hz (200 ms)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot_and_labels)
        self.plot_timer.start(200)

    def init_thermal_architecture(self):
        """ Inicializa el modelo nodal continuo con inercia térmica física """
        self.T_panels_tip = 48.0 # Punta de los paneles solares
        self.T_top = 34.0        # Cara superior (+Z) y raíz de bisagras
        self.T_upper = 26.0      # Chasis superior (Z: +50 a +140 mm)
        self.T_mid = 21.0        # Chasis medio (Z: -30 a +50 mm)
        self.T_lower = 16.0      # Chasis inferior (Z: -120 a -30 mm)
        self.T_bottom = 12.0     # Cara inferior (-Z nadir)

        self.sources = [
            HeatSourceNode("OBC (STM32F429)", "Aviónica", [0, 0, 60], power=1.5, max_power=4.0),
            HeatSourceNode("SDR Payload (20MHz)", "Carga Útil", [0, 0, 10], power=0.0, max_power=6.0),
            HeatSourceNode("TT&C Tx (S-Band / UHF)", "Comms", [0, 0, -40], power=0.5, max_power=5.0),
            HeatSourceNode("EPS Módulo & Buck 5V", "Potencia", [0, 0, -90], power=0.7, max_power=3.0),
            HeatSourceNode("Batería Li-Ion (20Wh)", "Almacenamiento", [0, 0, -135], power=0.3, max_power=3.0),
            HeatSourceNode("AOCS Reaction Wheels", "Actuadores", [0, 0, 120], power=0.6, max_power=3.5),
            HeatSourceNode("GNSS Receptor Científico", "Carga Útil", [0, 0, 100], power=0.0, max_power=2.0),
        ]

        self.thermistors = [
            ThermistorSensor("T01", "Panel Solar +X", "EPS", [230, 0, 170], -30, 70),
            ThermistorSensor("T02", "Panel Solar -X", "EPS", [-230, 0, 170], -30, 70),
            ThermistorSensor("T03", "Panel Solar +Y", "EPS", [0, 230, 170], -30, 70),
            ThermistorSensor("T04", "Panel Solar -Y", "EPS", [0, -230, 170], -30, 70),
            ThermistorSensor("T05", "Cara Superior +Z", "Estructura", [0, 0, 170], -20, 65),
            ThermistorSensor("T06", "Cara Inferior -Z", "Estructura", [0, 0, -170], -30, 50),
            ThermistorSensor("T07", "Lateral +X Chasis", "Estructura", [48, 0, 0], -20, 60),
            ThermistorSensor("T08", "Lateral -X Chasis", "Estructura", [-48, 0, 0], -20, 60),
            ThermistorSensor("T09", "Lateral +Y Chasis", "Estructura", [0, 48, 0], -20, 60),
            ThermistorSensor("T10", "Lateral -Y Chasis", "Estructura", [0, -48, 0], -20, 60),
            ThermistorSensor("T11", "OBC STM32 MCU", "OBDH", [15, 15, 60], -10, 55),
            ThermistorSensor("T12", "OBC Thermal Strap", "TCS", [40, 40, 60], -15, 55),
            ThermistorSensor("T13", "SDR LNA Front-End", "Payload", [15, -15, 10], -10, 60),
            ThermistorSensor("T14", "SDR Procesador DSP", "Payload", [-15, 15, 10], -10, 65),
            ThermistorSensor("T15", "TT&C Power Amp", "COMMS", [0, 20, -40], -10, 70),
            ThermistorSensor("T16", "EPS Regulador 5V", "EPS", [20, 0, -90], -10, 65),
            ThermistorSensor("T17", "Batería Celda 1", "EPS", [15, 15, -135], 0, 45),
            ThermistorSensor("T18", "Batería Celda 2", "EPS", [-15, -15, -135], 0, 45),
            ThermistorSensor("T19", "AOCS Ruedas Inercia", "AOCS", [0, 0, 120], -15, 55),
            ThermistorSensor("T20", "Chasis Core Al6061", "Estructura", [0, 0, 0], -20, 60)
        ]

    def init_gui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { border: 1px solid #30363d; border-radius: 5px; margin-top: 8px; padding-top: 6px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #58a6ff; font-size: 11px; }
            QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; font-size: 11px; color: #c9d1d9; }
            QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
            QPushButton:pressed { background-color: #1f6feb; color: white; }
            QTabWidget::pane { border: 1px solid #30363d; border-radius: 4px; background-color: #161b22; }
            QTabBar::tab { background: #0d1117; border: 1px solid #30363d; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 6px 12px; margin-right: 2px; color: #8b949e; font-size: 11px; }
            QTabBar::tab:selected { background: #161b22; border-color: #58a6ff; color: #58a6ff; font-weight: bold; }
            QTableWidget { background-color: #0d1117; gridline-color: #30363d; border: 1px solid #30363d; border-radius: 4px; font-size: 11px; }
            QHeaderView::section { background-color: #161b22; color: #58a6ff; padding: 3px; border: 1px solid #30363d; font-weight: bold; font-size: 11px; }
            QSlider::groove:horizontal { height: 4px; background: #21262d; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }
            QSlider::handle:horizontal { background: #58a6ff; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }
        """)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # 1. BARRA SUPERIOR COMPACTA (38px de altura fija)
        header_frame = QFrame()
        header_frame.setFixedHeight(38)
        header_frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 4px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 2, 8, 2)
        header_layout.setSpacing(6)

        title_lbl = QLabel("<b>S-ART 3U</b> | TCS Console")
        title_lbl.setStyleSheet("font-size: 13px; color: #58a6ff; font-weight: bold;")
        header_layout.addWidget(title_lbl)

        self.orbit_badge = QLabel("SOL (1361 W/m²)")
        self.orbit_badge.setStyleSheet("background-color: #d29922; color: #0d1117; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.orbit_badge)

        self.power_badge = QLabel("Potencia: 2.7 W")
        self.power_badge.setStyleSheet("background-color: #21262d; color: #58a6ff; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.power_badge)

        # Badge dedicado de radiación al vacío y temperatura equivalente
        self.rad_badge = QLabel("Q_rad: 54.2 W | T_eq: 26.5°C")
        self.rad_badge.setStyleSheet("background-color: #21262d; color: #bc8cff; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.rad_badge)

        self.temp_max_badge = QLabel("Máx: 48.0°C")
        self.temp_max_badge.setStyleSheet("background-color: #21262d; color: #ff7b72; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.temp_max_badge)

        self.temp_min_badge = QLabel("Mín: 12.0°C")
        self.temp_min_badge.setStyleSheet("background-color: #21262d; color: #79c0ff; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.temp_min_badge)

        self.battery_safety_badge = QLabel("Batería: OK")
        self.battery_safety_badge.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        header_layout.addWidget(self.battery_safety_badge)

        header_layout.addStretch()

        header_layout.addWidget(QLabel("<font color='#8b949e'>Modos:</font>"))
        btn_nom = QPushButton("SDR On")
        btn_nom.clicked.connect(lambda: self.apply_preset_mode("SCIENCE"))
        header_layout.addWidget(btn_nom)

        btn_comms = QPushButton("TT&C Tx")
        btn_comms.clicked.connect(lambda: self.apply_preset_mode("COMMS"))
        header_layout.addWidget(btn_comms)

        btn_safe = QPushButton("Safe Mode")
        btn_safe.clicked.connect(lambda: self.apply_preset_mode("SAFE"))
        header_layout.addWidget(btn_safe)

        main_layout.addWidget(header_frame)

        # 2. CUERPO PRINCIPAL
        body_splitter = QSplitter(Qt.Horizontal)

        # Lado Izquierdo con Splitter Vertical
        left_splitter = QSplitter(Qt.Vertical)

        left_top_widget = QWidget()
        lt_layout = QVBoxLayout(left_top_widget)
        lt_layout.setContentsMargins(0, 0, 0, 0)
        lt_layout.setSpacing(2)

        self.view3d = CubeSat3DView(self)
        lt_layout.addWidget(self.view3d, stretch=1)

        # Controles rápidos de viewport
        view_ctrls = QWidget()
        view_ctrls.setFixedHeight(30)
        vc_layout = QHBoxLayout(view_ctrls)
        vc_layout.setContentsMargins(4, 2, 4, 2)
        vc_layout.setSpacing(4)
        
        vc_layout.addWidget(QLabel("Cámara:"))
        for name, el, az in [("Iso", 25, 50), ("Sol (+Z)", 89, 0), ("Tierra (-Z)", -89, 0), ("Lat X", 0, 0), ("Lat Y", 0, 90)]:
            b = QPushButton(name)
            b.setFixedHeight(20)
            b.clicked.connect(lambda _, e=el, a=az: self.view3d.setCameraPosition(elevation=e, azimuth=a))
            vc_layout.addWidget(b)

        vc_layout.addSpacing(10)
        # Toggles de Nodos y Etiquetas
        self.chk_sources = QCheckBox("Nodos Calor")
        self.chk_sources.setChecked(True)
        self.chk_sources.stateChanged.connect(self.on_toggle_sources)
        vc_layout.addWidget(self.chk_sources)

        self.chk_labels_sources = QCheckBox("Etiquetas (W)")
        self.chk_labels_sources.setChecked(True)
        self.chk_labels_sources.stateChanged.connect(self.on_toggle_source_labels)
        vc_layout.addWidget(self.chk_labels_sources)

        self.chk_therms = QCheckBox("Termistores")
        self.chk_therms.setChecked(True)
        self.chk_therms.stateChanged.connect(self.on_toggle_thermistors)
        vc_layout.addWidget(self.chk_therms)

        self.chk_labels_therms = QCheckBox("Etiquetas (°C)")
        self.chk_labels_therms.setChecked(True)
        self.chk_labels_therms.stateChanged.connect(self.on_toggle_therm_labels)
        vc_layout.addWidget(self.chk_labels_therms)

        vc_layout.addStretch()
        vc_layout.addWidget(QLabel("Escala:"))
        self.colorbar = ColorBarWidget(-30, 70)
        vc_layout.addWidget(self.colorbar)

        lt_layout.addWidget(view_ctrls)
        left_splitter.addWidget(left_top_widget)

        # Gráfico temporal optimizado
        self.plot_widget = pg.PlotWidget(title="Telemetría Térmica Temporal (Al6061)")
        self.plot_widget.setFixedHeight(140)
        self.plot_widget.addLegend(offset=(10, 5))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('left', '°C')
        self.plot_widget.setLabel('bottom', 's')
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(auto=True, mode='peak')
        
        self.curve_pan = self.plot_widget.plot(pen=pg.mkPen('#ff7b72', width=2), name="Paneles (+Z)")
        self.curve_top = self.plot_widget.plot(pen=pg.mkPen('#f0883e', width=2), name="Cara Sol")
        self.curve_obc = self.plot_widget.plot(pen=pg.mkPen('#58a6ff', width=2), name="OBC MCU")
        self.curve_bat = self.plot_widget.plot(pen=pg.mkPen('#e3b341', width=2), name="Batería")
        self.curve_bot = self.plot_widget.plot(pen=pg.mkPen('#79c0ff', width=2), name="Cara Tierra")
        
        left_splitter.addWidget(self.plot_widget)
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 1)

        body_splitter.addWidget(left_splitter)

        # Lado Derecho: Pestañas de configuración
        right_widget = QWidget()
        right_widget.setMinimumWidth(440)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        self.init_tab_hotspots()
        self.init_tab_telemetry()
        self.init_tab_orbit_physics()

        body_splitter.addWidget(right_widget)
        body_splitter.setStretchFactor(0, 3)
        body_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(body_splitter)

    def init_tab_hotspots(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        top_group = QGroupBox("Puntos Calientes / Fuentes Internas")
        top_layout = QVBoxLayout(top_group)

        row_sel = QHBoxLayout()
        row_sel.addWidget(QLabel("Elemento:"))
        self.combo_source = QComboBox()
        for src in self.sources:
            self.combo_source.addItem(f"{src.name} [{src.subsystem}]")
        self.combo_source.currentIndexChanged.connect(self.on_select_source_combo)
        row_sel.addWidget(self.combo_source)
        top_layout.addLayout(row_sel)

        pos_box = QGroupBox("Coordenadas Internas (mm)")
        pos_layout = QHBoxLayout(pos_box)
        self.spin_x = QDoubleSpinBox(); self.spin_x.setRange(-350, 350); self.spin_x.setSingleStep(5)
        self.spin_x.valueChanged.connect(self.on_coord_changed)
        pos_layout.addWidget(QLabel("X:")); pos_layout.addWidget(self.spin_x)

        self.spin_y = QDoubleSpinBox(); self.spin_y.setRange(-350, 350); self.spin_y.setSingleStep(5)
        self.spin_y.valueChanged.connect(self.on_coord_changed)
        pos_layout.addWidget(QLabel("Y:")); pos_layout.addWidget(self.spin_y)

        self.spin_z = QDoubleSpinBox(); self.spin_z.setRange(-170, 170); self.spin_z.setSingleStep(5)
        self.spin_z.valueChanged.connect(self.on_coord_changed)
        pos_layout.addWidget(QLabel("Z:")); pos_layout.addWidget(self.spin_z)
        top_layout.addWidget(pos_box)

        btn_row = QHBoxLayout()
        self.btn_pick = QPushButton("📍 Picar en 3D")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(self.toggle_picking_mode)
        btn_row.addWidget(self.btn_pick)

        self.combo_preset_pos = QComboBox()
        self.combo_preset_pos.addItems([
            "Alinear con Preset...", "Placa OBC (Z=+60)", "Payload SDR (Z=+10)", 
            "Transmisor TT&C (Z=-40)", "Placa EPS (Z=-90)", "Batería Li-Ion (Z=-135)",
            "Panel Solar +X", "Panel Solar -X", "Cara Superior Sol (+Z)", "Cara Tierra (-Z)"
        ])
        self.combo_preset_pos.currentIndexChanged.connect(self.apply_preset_position)
        btn_row.addWidget(self.combo_preset_pos)
        top_layout.addLayout(btn_row)

        pwr_group = QGroupBox("Potencia Térmica Disipada")
        pwr_layout = QVBoxLayout(pwr_group)
        row_pwr = QHBoxLayout()
        self.chk_source_enable = QCheckBox("Habilitado")
        self.chk_source_enable.setChecked(True)
        self.chk_source_enable.stateChanged.connect(self.on_toggle_source_enable)
        row_pwr.addWidget(self.chk_source_enable)

        self.lbl_power_val = QLabel("1.5 W")
        self.lbl_power_val.setStyleSheet("font-weight: bold; color: #ff7b72;")
        row_pwr.addStretch()
        row_pwr.addWidget(self.lbl_power_val)
        pwr_layout.addLayout(row_pwr)

        self.slider_power = QSlider(Qt.Horizontal)
        self.slider_power.setRange(0, 100) # 0 a 10 W
        self.slider_power.setValue(15)
        self.slider_power.valueChanged.connect(self.on_power_slider_changed)
        pwr_layout.addWidget(self.slider_power)

        top_layout.addWidget(pwr_group)
        layout.addWidget(top_group)

        layout.addWidget(QLabel("<b>Fuentes de Calor Activas (Click para seleccionar):</b>"))
        self.table_sources = QTableWidget(len(self.sources), 4)
        self.table_sources.setHorizontalHeaderLabels(["Nombre", "Subsis.", "Potencia", "Pos (X,Y,Z)"])
        self.table_sources.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_sources.cellClicked.connect(self.on_table_source_clicked)
        self.update_sources_table()
        layout.addWidget(self.table_sources)

        self.tabs.addTab(tab, "Puntos Calientes")
        self.on_select_source_combo(0)

    def init_tab_telemetry(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Telemetría de Termistores (Click para seleccionar y resaltar en 3D):</b>"))
        self.table_therms = QTableWidget(len(self.thermistors), 5)
        self.table_therms.setHorizontalHeaderLabels(["ID", "Ubicación", "Subsistema", "Temp (°C)", "Estado"])
        self.table_therms.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_therms.cellClicked.connect(self.on_table_therm_clicked)
        for i, t in enumerate(self.thermistors):
            self.table_therms.setItem(i, 0, QTableWidgetItem(t.id))
            self.table_therms.setItem(i, 1, QTableWidgetItem(t.name))
            self.table_therms.setItem(i, 2, QTableWidgetItem(t.subsystem))
            self.table_therms.setItem(i, 3, QTableWidgetItem("20.0 °C"))
            b = QTableWidgetItem("NORMAL")
            b.setForeground(QColor("#3fb950"))
            self.table_therms.setItem(i, 4, b)
        layout.addWidget(self.table_therms)

        layout.addWidget(QLabel("<b>Trama Serial S-ART (NMEA):</b>"))
        self.txt_serial = QLabel("$SART,0,20.0,...*")
        self.txt_serial.setWordWrap(True)
        self.txt_serial.setStyleSheet("font-family: Consolas, monospace; background-color: #0d1117; color: #3fb950; padding: 6px; border: 1px solid #30363d; border-radius: 4px;")
        layout.addWidget(self.txt_serial)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copiar Trama")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_serial.text()))
        btn_row.addWidget(btn_copy)

        btn_csv = QPushButton("💾 Exportar CSV")
        btn_csv.clicked.connect(self.export_csv_telemetry)
        btn_row.addWidget(btn_csv)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Termistores & Serial")

    def init_tab_orbit_physics(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # Cuadro de Balance Radiativo al Vacío
        rad_group = QGroupBox("Balance Radiativo al Vacío Espacial (Stefan-Boltzmann)")
        rad_layout = QVBoxLayout(rad_group)
        self.lbl_q_rad_info = QLabel("• Potencia Irradiada al Espacio: <b>54.2 W</b>")
        self.lbl_t_rad_info = QLabel("• Temperatura Radiativa Equivalente: <b>+26.5 °C</b>")
        self.lbl_q_net_info = QLabel("• Flujo Térmico Neto: <b>+8.1 W</b> (Calentándose)")
        rad_layout.addWidget(self.lbl_q_rad_info)
        rad_layout.addWidget(self.lbl_t_rad_info)
        rad_layout.addWidget(self.lbl_q_net_info)
        layout.addWidget(rad_group)

        orb_group = QGroupBox("Modo de Iluminación Orbital")
        orb_layout = QVBoxLayout(orb_group)
        self.radio_orb_auto = QRadioButton("Órbita Kepleriana Real (Día / Noche continua)")
        self.radio_orb_auto.setChecked(True)
        self.radio_orb_auto.toggled.connect(lambda: self.set_eclipse_mode("AUTO"))
        orb_layout.addWidget(self.radio_orb_auto)

        self.radio_orb_sun = QRadioButton("Exposición Solar Continua (Hot Case)")
        self.radio_orb_sun.toggled.connect(lambda: self.set_eclipse_mode("FORCE_SUN"))
        orb_layout.addWidget(self.radio_orb_sun)

        self.radio_orb_ecl = QRadioButton("Eclipse Permanente (Cold Case)")
        self.radio_orb_ecl.toggled.connect(lambda: self.set_eclipse_mode("FORCE_ECLIPSE"))
        orb_layout.addWidget(self.radio_orb_ecl)

        alt_row = QHBoxLayout()
        alt_row.addWidget(QLabel("Altitud de Órbita:"))
        self.lbl_alt = QLabel("440 km (LEO)")
        alt_row.addWidget(self.lbl_alt)
        orb_layout.addLayout(alt_row)

        self.slider_alt = QSlider(Qt.Horizontal)
        self.slider_alt.setRange(250, 700)
        self.slider_alt.setValue(440)
        self.slider_alt.valueChanged.connect(self.on_alt_slider_changed)
        orb_layout.addWidget(self.slider_alt)
        layout.addWidget(orb_group)

        heat_group = QGroupBox("Calentador Activo Kapton (SRS-TCS-SART-100)")
        heat_layout = QVBoxLayout(heat_group)
        self.chk_battery_heater = QCheckBox("Termostato Autónomo Batería (Calentar si T < 2°C)")
        self.chk_battery_heater.setChecked(True)
        heat_layout.addWidget(self.chk_battery_heater)
        layout.addWidget(heat_group)

        speed_group = QGroupBox("Velocidad de Simulación")
        sp_layout = QHBoxLayout(speed_group)
        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_pause.clicked.connect(self.toggle_pause)
        sp_layout.addWidget(self.btn_pause)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["1x Real", "10x", "50x Rápida", "100x Acelerada"])
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.currentIndexChanged.connect(self.on_speed_changed)
        sp_layout.addWidget(self.combo_speed)
        layout.addWidget(speed_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Entorno & Órbita")

    def load_full_cad_model(self):
        """ Carga la geometría 3D y configura etiquetas 3D interactivas """
        stl_path = 'S-art_sd.stl'
        if not os.path.exists(stl_path): return

        stl_data = stl_mesh.Mesh.from_file(stl_path)
        vectors = stl_data.vectors
        self.N_faces = len(vectors)

        self.vectors_centered = (vectors - np.array([0.0, 0.0, 170.0], dtype=np.float32)).astype(np.float32)
        self.face_centers = self.vectors_centered.mean(axis=1)

        x = self.face_centers[:, 0]
        y = self.face_centers[:, 1]
        z = self.face_centers[:, 2]
        self.r_xy = np.sqrt(x**2 + y**2)
        self.is_solar_panel = (z > 120.0) & (self.r_xy > 50.0)
        self.panel_xi = np.clip((self.r_xy - 50.0) / 295.0, 0.0, 1.0)
        self.z_norm = np.clip((z + 170.0) / 340.0, 0.0, 1.0)

        vertices = self.vectors_centered.reshape(self.N_faces * 3, 3)
        faces = np.arange(self.N_faces * 3, dtype=np.uint32).reshape(self.N_faces, 3)
        self.meshdata = gl.MeshData(vertexes=vertices, faces=faces)
        self.meshdata._faceNormals = stl_data.normals

        init_colors = np.full((self.N_faces, 4), [0.2, 0.4, 0.7, 1.0], dtype=np.float32)
        self.meshdata.setFaceColors(init_colors)

        self.mesh_item = gl.GLMeshItem(
            meshdata=self.meshdata,
            smooth=False,
            drawEdges=False,
            shader='solar_shaded'
        )
        self.view3d.addItem(self.mesh_item)

        self.axis_item = gl.GLAxisItem()
        self.axis_item.setSize(220, 220, 220)
        self.view3d.addItem(self.axis_item)

        # Marcadores 3D
        src_pos = np.array([s.pos for s in self.sources], dtype=np.float32)
        self.source_markers = gl.GLScatterPlotItem(pos=src_pos, color=(1.0, 0.5, 0.0, 1.0), size=14)
        self.view3d.addItem(self.source_markers)

        th_pos = np.array([t.pos for t in self.thermistors], dtype=np.float32)
        self.therm_markers = gl.GLScatterPlotItem(pos=th_pos, color=(0.0, 0.9, 0.4, 1.0), size=10)
        self.view3d.addItem(self.therm_markers)

        # Marcadores de Resaltado (Highlight Selection)
        self.highlight_source_marker = gl.GLScatterPlotItem(pos=np.array([self.sources[0].pos]), color=(1.0, 1.0, 0.2, 1.0), size=24)
        self.view3d.addItem(self.highlight_source_marker)

        self.highlight_therm_marker = gl.GLScatterPlotItem(pos=np.empty((0, 3)), color=(0.2, 1.0, 1.0, 1.0), size=20)
        self.view3d.addItem(self.highlight_therm_marker)

        # ==================== ETIQUETAS 3D (GLTextItem) ====================
        label_font = QFont('Segoe UI', 9, QFont.Bold)

        # Etiquetas de Fuentes de Calor (Potencia en W)
        for s in self.sources:
            t_pos = s.pos + np.array([5.0, -10.0, 12.0], dtype=np.float32)
            lbl = gl.GLTextItem(pos=t_pos, text=f"{s.name}: {s.power:.1f}W", color=QColor('#ffa657'), font=label_font)
            self.view3d.addItem(lbl)
            s.label_item = lbl

        # Etiquetas de Termistores (Temperatura en °C)
        for t in self.thermistors:
            t_pos = t.pos + np.array([5.0, 5.0, 10.0], dtype=np.float32)
            lbl = gl.GLTextItem(pos=t_pos, text=f"{t.id}: 20.0°C", color=QColor('#7ee787'), font=label_font)
            self.view3d.addItem(lbl)
            t.label_item = lbl

        self.cmap = create_colormap(-30, 70, steps=1000)
        self.last_render_temp = -999.0

    # ==================== GESTIÓN DE VISIBILIDAD DE MARCADORES Y ETIQUETAS ====================
    def on_toggle_thermistors(self, state):
        visible = (state == Qt.Checked)
        if visible:
            th_pos = np.array([t.pos for t in self.thermistors], dtype=np.float32)
            self.therm_markers.setData(pos=th_pos)
            self.therm_markers.setVisible(True)
        else:
            self.therm_markers.setData(pos=np.empty((0, 3), dtype=np.float32))
            self.therm_markers.setVisible(False)
        self.view3d.update()

    def on_toggle_therm_labels(self, state):
        visible = (state == Qt.Checked)
        for t in self.thermistors:
            if t.label_item:
                t.label_item.setVisible(visible)
        self.view3d.update()

    def on_toggle_sources(self, state):
        visible = (state == Qt.Checked)
        if visible:
            src_pos = np.array([s.pos for s in self.sources if s.enabled], dtype=np.float32)
            self.source_markers.setData(pos=src_pos)
            self.source_markers.setVisible(True)
        else:
            self.source_markers.setData(pos=np.empty((0, 3), dtype=np.float32))
            self.source_markers.setVisible(False)
        self.view3d.update()

    def on_toggle_source_labels(self, state):
        visible = (state == Qt.Checked)
        for s in self.sources:
            if s.label_item:
                s.label_item.setVisible(visible)
        self.view3d.update()

    # ==================== SELECCIÓN INTERACTIVA DE ELEMENTOS ====================
    def on_select_source_combo(self, idx):
        if idx < 0 or idx >= len(self.sources): return
        self.selected_source_idx = idx
        src = self.sources[idx]

        self.spin_x.blockSignals(True); self.spin_x.setValue(src.pos[0]); self.spin_x.blockSignals(False)
        self.spin_y.blockSignals(True); self.spin_y.setValue(src.pos[1]); self.spin_y.blockSignals(False)
        self.spin_z.blockSignals(True); self.spin_z.setValue(src.pos[2]); self.spin_z.blockSignals(False)
        self.chk_source_enable.blockSignals(True); self.chk_source_enable.setChecked(src.enabled); self.chk_source_enable.blockSignals(False)
        self.slider_power.blockSignals(True); self.slider_power.setValue(int(src.power * 10)); self.slider_power.blockSignals(False)
        self.lbl_power_val.setText(f"{src.power:.1f} W")

        # Resaltar en 3D
        if hasattr(self, 'highlight_source_marker'):
            self.highlight_source_marker.setData(pos=np.array([src.pos]))
            self.highlight_source_marker.setVisible(True)

    def on_table_source_clicked(self, row, col):
        self.combo_source.setCurrentIndex(row)

    def on_table_therm_clicked(self, row, col):
        if row < 0 or row >= len(self.thermistors): return
        self.selected_therm_idx = row
        th = self.thermistors[row]
        if hasattr(self, 'highlight_therm_marker'):
            self.highlight_therm_marker.setData(pos=np.array([th.pos]))
            self.highlight_therm_marker.setVisible(True)
            self.view3d.update()

    def on_coord_changed(self):
        idx = self.combo_source.currentIndex()
        if idx < 0 or idx >= len(self.sources): return
        src = self.sources[idx]
        src.pos = np.array([self.spin_x.value(), self.spin_y.value(), self.spin_z.value()], dtype=np.float32)
        if src.label_item:
            src.label_item.setData(pos=src.pos + np.array([5.0, -10.0, 12.0], dtype=np.float32))
        self.update_sources_table()
        self.on_toggle_sources(self.chk_sources.checkState())
        if hasattr(self, 'highlight_source_marker'):
            self.highlight_source_marker.setData(pos=np.array([src.pos]))

    def apply_preset_position(self, p_idx):
        if p_idx <= 0: return
        presets = {
            1: (0, 0, 60), 2: (0, 0, 10), 3: (0, 0, -40), 4: (0, 0, -90),
            5: (0, 0, -135), 6: (220, 0, 170), 7: (-220, 0, 170), 8: (0, 0, 170), 9: (0, 0, -170)
        }
        if p_idx in presets:
            pos = presets[p_idx]
            self.spin_x.setValue(pos[0])
            self.spin_y.setValue(pos[1])
            self.spin_z.setValue(pos[2])
        self.combo_preset_pos.setCurrentIndex(0)

    def toggle_picking_mode(self, enabled):
        self.picking_mode = enabled
        if enabled:
            self.btn_pick.setStyleSheet("background-color: #f0883e; color: #0d1117; font-weight: bold;")
            self.btn_pick.setText("Haz Click en el CAD...")
        else:
            self.btn_pick.setStyleSheet("")
            self.btn_pick.setText("📍 Picar en 3D")

    def handle_3d_pick(self, screen_pos):
        w = self.view3d.width(); h = self.view3d.height()
        region = (0, 0, w, h)
        view_mat = self.view3d.viewMatrix()
        proj_mat = self.view3d.projectionMatrix(region, region)
        M = np.array((proj_mat * view_mat).data()).reshape(4,4)

        sample_pts = self.face_centers[::10]
        pts4 = np.hstack([sample_pts, np.ones((len(sample_pts), 1), dtype=np.float32)])
        pts_clip = np.dot(pts4, M)

        valid = pts_clip[:, 3] > 0
        if not np.any(valid): return

        pts_ndc = pts_clip[valid, :3] / pts_clip[valid, 3:4]
        x = (pts_ndc[:, 0] + 1) * w / 2.0
        y = (1 - pts_ndc[:, 1]) * h / 2.0
        dist2 = (x - screen_pos.x())**2 + (y - screen_pos.y())**2
        candidates = np.where(valid & (dist2 < 400.0))[0]

        if len(candidates) > 0:
            best_idx = candidates[np.argmin(pts_ndc[candidates, 2])]
            picked_pt = sample_pts[best_idx]
            self.spin_x.setValue(float(picked_pt[0]))
            self.spin_y.setValue(float(picked_pt[1]))
            self.spin_z.setValue(float(picked_pt[2]))

        self.btn_pick.setChecked(False)

    def on_toggle_source_enable(self, state):
        idx = self.combo_source.currentIndex()
        if idx >= 0:
            self.sources[idx].enabled = (state == Qt.Checked)
            self.update_sources_table()
            self.on_toggle_sources(self.chk_sources.checkState())

    def on_power_slider_changed(self, val):
        idx = self.combo_source.currentIndex()
        if idx >= 0:
            power = val / 10.0
            self.sources[idx].power = power
            self.lbl_power_val.setText(f"{power:.1f} W")
            self.update_sources_table()

    def update_sources_table(self):
        for i, src in enumerate(self.sources):
            pwr_str = f"{src.power:.1f} W" if src.enabled else "OFF"
            pos_str = f"({int(src.pos[0])}, {int(src.pos[1])}, {int(src.pos[2])})"
            self.table_sources.setItem(i, 0, QTableWidgetItem(src.name))
            self.table_sources.setItem(i, 1, QTableWidgetItem(src.subsystem))
            self.table_sources.setItem(i, 2, QTableWidgetItem(pwr_str))
            self.table_sources.setItem(i, 3, QTableWidgetItem(pos_str))

    def set_eclipse_mode(self, mode):
        self.eclipse_mode = mode

    def on_alt_slider_changed(self, val):
        self.altitude_km = float(val)
        self.lbl_alt.setText(f"{val} km (LEO)")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("▶ Reanudar" if self.is_paused else "⏸ Pausar")

    def on_speed_changed(self, idx):
        speeds = [1.0, 10.0, 50.0, 100.0]
        self.sim_speed = speeds[idx]

    def apply_preset_mode(self, mode):
        if mode == "SCIENCE":
            for s in self.sources:
                if "SDR" in s.name: s.power = 4.5; s.enabled = True
                if "OBC" in s.name: s.power = 2.5; s.enabled = True
        elif mode == "COMMS":
            for s in self.sources:
                if "TT&C" in s.name: s.power = 4.2; s.enabled = True
                if "SDR" in s.name: s.power = 0.0; s.enabled = False
        elif mode == "SAFE":
            for s in self.sources:
                if "OBC" in s.name: s.power = 0.8; s.enabled = True
                if "SDR" in s.name: s.power = 0.0; s.enabled = False
                if "TT&C" in s.name: s.power = 0.4; s.enabled = True
        self.on_select_source_combo(self.combo_source.currentIndex())
        self.update_sources_table()

    def export_csv_telemetry(self):
        path = "telemetry_log.csv"
        with open(path, "w") as f:
            f.write("Sensor_ID,Nombre,Subsistema,Temperatura_C\n")
            for t in self.thermistors:
                f.write(f"{t.id},{t.name},{t.subsystem},{t.temp:.2f}\n")
        print(f"Telemetría exportada a {path}")

    # ==================== FÍSICA TÉRMICA RIGUROSA EN LEO ====================
    def update_physics(self):
        """ Simulación física diferencial continua y rigurosa en LEO (20 Hz) """
        if self.is_paused: return
        dt = 0.05 * self.sim_speed
        self.sim_time += dt

        # 1. Geometría Orbital Kepleriana (440 km LEO, Periodo ~93.3 min)
        r = 6371.0 + self.altitude_km
        period = 2.0 * np.pi * np.sqrt((r * 1000.0)**3 / 3.986e14)
        phase = (self.sim_time % period) / period
        theta = np.arcsin(6371.0 / r)
        eclipse_fraction = theta / np.pi

        if self.eclipse_mode == "FORCE_SUN":
            target_sun = 1.0
            self.in_eclipse = False
        elif self.eclipse_mode == "FORCE_ECLIPSE":
            target_sun = 0.0
            self.in_eclipse = True
        else:
            in_shadow = (phase > (1.0 - eclipse_fraction / 2.0)) or (phase < (eclipse_fraction / 2.0))
            self.in_eclipse = in_shadow
            target_sun = 0.0 if in_shadow else 1.0

        # Transición orbital realista al cruzar el terminador solar (~18s)
        self.sun_intensity += (target_sun - self.sun_intensity) * min(1.0, dt / 18.0)

        # Actualizar indicador solar
        if self.sun_intensity > 0.5:
            self.orbit_badge.setText(f"SOL ({int(1361*self.sun_intensity)} W/m²)")
            self.orbit_badge.setStyleSheet("background-color: #d29922; color: #0d1117; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        else:
            self.orbit_badge.setText("ECLIPSE (Sombra LEO)")
            self.orbit_badge.setStyleSheet("background-color: #388bfd; color: white; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")

        # RADIACIÓN Y FLUJOS TÉRMICOS:
        q_earth_ir_flux = 237.0 # Emisión infrarroja continua de la Tierra (W/m2)
        
        # 1. PANELES SOLARES
        q_solar_panels = self.sun_intensity * 1361.0 * 0.12 * 0.90 * (1.0 - 0.28) # ~105.8 W en sol
        q_earth_ir_panels = 0.85 * 0.12 * q_earth_ir_flux * 0.15 # ~3.6 W
        t_pan_k = self.T_panels_tip + 273.15
        q_rad_panels = 0.85 * SIGMA_SB * 0.24 * (t_pan_k**4)
        q_cond_panels_to_top = 0.8 * (self.T_top - self.T_panels_tip)
        dT_panels = (q_solar_panels + q_earth_ir_panels + q_cond_panels_to_top - q_rad_panels) * dt / 716.8
        self.T_panels_tip += dT_panels

        # 2. CARA SUPERIOR (+Z)
        q_solar_top = self.sun_intensity * 1361.0 * 0.01 * 0.80 # ~10.9 W
        t_top_k = self.T_top + 273.15
        q_rad_top = 0.80 * SIGMA_SB * 0.01 * (t_top_k**4)
        q_cond_top_to_upper = 1.6 * (self.T_upper - self.T_top)
        dT_top = (q_solar_top - q_rad_top - q_cond_panels_to_top + q_cond_top_to_upper) * dt / 360.0
        self.T_top += dT_top

        # 3. CHASIS SUPERIOR (Z: +50 a +140 mm)
        p_upper_sources = sum(s.power for s in self.sources if s.enabled and s.pos[2] > 40)
        t_upper_k = self.T_upper + 273.15
        q_rad_upper = 0.70 * SIGMA_SB * 0.02 * (t_upper_k**4)
        q_cond_up_to_mid = 2.0 * (self.T_mid - self.T_upper)
        dT_upper = (p_upper_sources - q_rad_upper - q_cond_top_to_upper + q_cond_up_to_mid) * dt / 1075.0
        self.T_upper += dT_upper

        # 4. CHASIS MEDIO (Z: -30 a +50 mm)
        p_mid_sources = sum(s.power for s in self.sources if s.enabled and -30 <= s.pos[2] <= 40)
        t_mid_k = self.T_mid + 273.15
        q_rad_mid = 0.70 * SIGMA_SB * 0.02 * (t_mid_k**4)
        q_cond_mid_to_low = 2.0 * (self.T_lower - self.T_mid)
        dT_mid = (p_mid_sources - q_rad_mid - q_cond_up_to_mid + q_cond_mid_to_low) * dt / 1344.0
        self.T_mid += dT_mid

        # 5. CHASIS INFERIOR (Z: -120 a -30 mm)
        p_low_sources = sum(s.power for s in self.sources if s.enabled and s.pos[2] < -30)
        
        # Termostato autónomo Kapton para Batería
        if self.chk_battery_heater.isChecked() and self.T_lower < 2.0:
            p_low_sources += 2.0
            self.battery_safety_badge.setText("Heater: ACTIVO (2W)")
            self.battery_safety_badge.setStyleSheet("background-color: #d29922; color: #0d1117; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        else:
            self.battery_safety_badge.setText("Batería: OK")
            self.battery_safety_badge.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 2px 8px; border-radius: 3px; font-size: 11px;")

        t_lower_k = self.T_lower + 273.15
        q_rad_lower = 0.70 * SIGMA_SB * 0.02 * (t_lower_k**4)
        q_cond_low_to_bot = 1.6 * (self.T_bottom - self.T_lower)
        dT_lower = (p_low_sources - q_rad_lower - q_cond_mid_to_low + q_cond_low_to_bot) * dt / 1612.0
        self.T_lower += dT_lower

        # 6. CARA INFERIOR (-Z Nadir)
        q_earth_ir_bot = 0.85 * 0.01 * q_earth_ir_flux * 0.70
        t_bot_k = self.T_bottom + 273.15
        q_rad_bot = 0.85 * SIGMA_SB * 0.01 * (t_bot_k**4)
        dT_bottom = (q_earth_ir_bot - q_rad_bot - q_cond_low_to_bot) * dt / 270.0
        self.T_bottom += dT_bottom

        # ==================== CÁLCULO DE RADIACIÓN AL VACÍO (STEFAN-BOLTZMANN) ====================
        # Potencia total irradiada al espacio por todas las superficies
        self.q_rad_total = float(q_rad_panels + q_rad_top + q_rad_upper + q_rad_mid + q_rad_lower + q_rad_bot)
        
        # Potencia total absorbida / generada
        total_p_internal = sum(s.power for s in self.sources if s.enabled)
        total_q_in = q_solar_panels + q_solar_top + q_earth_ir_panels + q_earth_ir_bot + total_p_internal
        self.q_net = total_q_in - self.q_rad_total

        # Temperatura equivalente de radiación al vacío (temperatura de cuerpo negro equivalente)
        t_eq_k = (self.q_rad_total / (EPSILON_AVG * SIGMA_SB * TOTAL_RAD_AREA))**0.25
        self.t_rad_equiv = float(t_eq_k - 273.15)

        # Actualizar indicadores de radiación
        net_sign = "+" if self.q_net >= 0 else ""
        self.rad_badge.setText(f"Q_rad: {self.q_rad_total:.1f}W | T_eq(vacío): {self.t_rad_equiv:.1f}°C | Q_net: {net_sign}{self.q_net:.1f}W")
        self.lbl_q_rad_info.setText(f"• Potencia Irradiada al Espacio (Q_rad): <b>{self.q_rad_total:.1f} W</b>")
        self.lbl_t_rad_info.setText(f"• Temperatura Radiativa de Equilibrio: <b>{self.t_rad_equiv:.1f} °C</b>")
        estado_net = "Calentándose" if self.q_net > 0 else "Enfriándose al vacío"
        self.lbl_q_net_info.setText(f"• Flujo Térmico Neto (Q_net): <b>{net_sign}{self.q_net:.1f} W</b> ({estado_net})")

        # 7. LECTURAS PRECISAS DE LOS 20 TERMISTORES (CON ELEVACIÓN TÉRMICA LOCAL REAL)
        # Resistencia térmica efectiva de montaje de componentes: R_th = 5.5 K/W
        # Esto genera incrementos locales de 15°C a 25°C cuando un componente consume 3-5W
        R_TH_LOCAL = 5.5 # K / W
        t_readings = []

        for i, th in enumerate(self.thermistors):
            pz = th.pos[2]
            pr = np.sqrt(th.pos[0]**2 + th.pos[1]**2)

            if th.name.startswith("Panel"):
                xi = np.clip((pr - 50.0) / 295.0, 0.0, 1.0)
                t_base = self.T_top + xi * (self.T_panels_tip - self.T_top)
            elif pz > 140:
                t_base = self.T_top
            elif pz > 50:
                a_z = (pz - 50.0) / 90.0
                t_base = (1.0 - a_z) * self.T_mid + a_z * self.T_upper
            elif pz > -30:
                a_z = (pz + 30.0) / 80.0
                t_base = (1.0 - a_z) * self.T_lower + a_z * self.T_mid
            elif pz > -140:
                a_z = (pz + 140.0) / 110.0
                t_base = (1.0 - a_z) * self.T_bottom + a_z * self.T_lower
            else:
                t_base = self.T_bottom

            # Conducción en el aluminio 6061 por fuentes activas (R_th * P * decay)
            t_local = 0.0
            for s in self.sources:
                if s.enabled and s.power > 0:
                    d = np.linalg.norm(th.pos - s.pos)
                    # Elevación térmica local real
                    t_local += s.power * R_TH_LOCAL * np.exp(-d / 40.0)

            t_final = t_base + t_local
            th.temp = float(t_final)
            t_readings.append(t_final)

            self.table_therms.item(i, 3).setText(f"{t_final:.1f} °C")
            b = self.table_therms.item(i, 4)
            if t_final > th.t_max:
                b.setText("CALOR"); b.setForeground(QColor("#ff7b72"))
            elif t_final < th.t_min:
                b.setText("FRÍO"); b.setForeground(QColor("#79c0ff"))
            else:
                b.setText("NORMAL"); b.setForeground(QColor("#3fb950"))

        # Métricas de encabezado
        max_t = max(t_readings); min_t = min(t_readings)
        self.power_badge.setText(f"Potencia: {total_p_internal:.1f} W")
        self.temp_max_badge.setText(f"Máx: {max_t:.1f}°C")
        self.temp_min_badge.setText(f"Mín: {min_t:.1f}°C")

        # Trama serial emulada
        telem_str = f"$SART,{int(self.sim_time)}"
        for t in self.thermistors: telem_str += f",{t.temp:.1f}"
        telem_str += "*"
        self.txt_serial.setText(telem_str)

        # Actualizar búfer circular para el gráfico
        if self.plot_count < self.plot_capacity:
            idx = self.plot_count
            self.plot_count += 1
        else:
            self.buf_time[:-1] = self.buf_time[1:]
            self.buf_pan[:-1] = self.buf_pan[1:]
            self.buf_top[:-1] = self.buf_top[1:]
            self.buf_obc[:-1] = self.buf_obc[1:]
            self.buf_bat[:-1] = self.buf_bat[1:]
            self.buf_bot[:-1] = self.buf_bot[1:]
            idx = self.plot_capacity - 1

        self.buf_time[idx] = self.sim_time
        self.buf_pan[idx] = self.T_panels_tip
        self.buf_top[idx] = self.T_top
        self.buf_obc[idx] = self.thermistors[10].temp
        self.buf_bat[idx] = self.thermistors[16].temp
        self.buf_bot[idx] = self.T_bottom

    # ==================== ACTUALIZACIÓN DE GRÁFICA Y ETIQUETAS (5 HZ) ====================
    def update_plot_and_labels(self):
        """ Actualiza la gráfica y las etiquetas 3D a 5 Hz de forma totalmente fluida """
        if self.is_paused: return

        # 1. Gráfica en búfer circular
        if self.plot_count >= 2:
            n = self.plot_count
            t_ax = self.buf_time[:n]
            self.curve_pan.setData(t_ax, self.buf_pan[:n])
            self.curve_top.setData(t_ax, self.buf_top[:n])
            self.curve_obc.setData(t_ax, self.buf_obc[:n])
            self.curve_bat.setData(t_ax, self.buf_bat[:n])
            self.curve_bot.setData(t_ax, self.buf_bot[:n])

        # 2. Actualizar texto de etiquetas 3D de Fuentes de Calor (Potencia en W)
        if self.chk_labels_sources.isChecked():
            for s in self.sources:
                if s.label_item:
                    pwr_text = f"{s.name}: {s.power:.1f}W" if s.enabled else f"{s.name}: OFF"
                    s.label_item.setData(text=pwr_text)

        # 3. Actualizar texto de etiquetas 3D de Termistores (Temperatura en °C)
        if self.chk_labels_therms.isChecked():
            for t in self.thermistors:
                if t.label_item:
                    t.label_item.setData(text=f"{t.id}: {t.temp:.1f}°C")

    # ==================== RENDERIZADO 3D CON PUNTOS CALIENTES VISIBLES (3 HZ) ====================
    def update_3d_render(self):
        """ Renderiza el modelo CAD con gradientes continuos y focos térmicos reales """
        if self.is_paused or not hasattr(self, 'meshdata'): return

        # 1. Gradiente a lo largo del chasis (Z: de -170mm a +170mm)
        T_body_grad = self.T_bottom + self.z_norm * (self.T_top - self.T_bottom)

        # 2. Transición matemáticamente suave a lo largo de los paneles solares
        T_panel_grad = self.T_top + self.panel_xi * (self.T_panels_tip - self.T_top)

        # 3. Combinar cuerpo y paneles
        T_faces = np.where(self.is_solar_panel, T_panel_grad, T_body_grad)

        # 4. Superponer puntos calientes con elevación térmica visible y realista:
        # Cada vatio eleva el centro del foco ~5.5°C con un radio de difusión de 35mm
        # Un SDR o TT&C activo a 4.5W producirá un punto brillante de +25°C perfectamente visible
        R_TH_CAD = 5.5 # K / W
        for s in self.sources:
            if s.enabled and s.power > 0:
                dx = self.face_centers[:, 0] - s.pos[0]
                dy = self.face_centers[:, 1] - s.pos[1]
                dz = self.face_centers[:, 2] - s.pos[2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                # Halo térmico visible y suave en el CAD
                T_faces += s.power * R_TH_CAD * np.exp(-dist / 35.0)

        # 5. Mapear a Colormap (-30°C a +70°C)
        idx = np.clip(((T_faces - (-30.0)) / 100.0 * 999.0).astype(np.int32), 0, 999)
        colors = self.cmap[idx]

        # 6. Actualizar buffer en OpenGL
        self.meshdata.setFaceColors(colors)
        self.mesh_item.meshDataChanged()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulatorApp()
    window.show()
    sys.exit(app.exec_())
