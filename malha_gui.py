import sys
import trimesh
import pymeshfix
import pymeshlab
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSizePolicy, QDoubleSpinBox, QMainWindow, QAction, QMenuBar, QInputDialog, QMessageBox, QFrame, QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem, GLLinePlotItem
from scipy.spatial import cKDTree


def trimesh_to_meshdata(mesh):
    # Converte uma malha trimesh para MeshData do pyqtgraph
    faces = mesh.faces
    vertices = mesh.vertices
    return MeshData(vertexes=vertices, faces=faces)


def create_glmeshitem(mesh, color=(0.5, 0.5, 1, 1)):
    meshdata = trimesh_to_meshdata(mesh)
    item = GLMeshItem(meshdata=meshdata, smooth=False, color=color, shader='shaded', drawEdges=True)
    return item


def centralizar_na_origem(mesh):
    # Move o centro do bounding box para a origem
    if mesh is None or not hasattr(mesh, 'bounding_box'):
        return mesh
    bbox = mesh.bounding_box
    center = bbox.centroid
    mesh.vertices -= center
    return mesh


class MeshRepairApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('🔧 Reparo de Malha 3D - STL/OBJ')
        self.resize(1400, 800)
        self.setStyleSheet(self.get_modern_stylesheet())
        
        # Configuração da paleta de cores
        self.setup_color_palette()
        
        self.mesh_original = None
        self.mesh_reparada = None
        
        # Widgets centrais
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Configuração dos widgets 3D com melhor aparência
        self.setup_3d_widgets()
        
        # Labels com melhor estilo
        self.setup_labels()
        
        # Botões principais com design moderno
        self.setup_main_buttons()
        
        # Layout principal com melhor organização
        self.setup_main_layout(central_widget)
        
        # Barra de menu com estilo melhorado
        self.setup_menu_bar()
        
        # Configuração das ações
        self.setup_actions()
        
        # Timer para atualizações visuais
        self.setup_timers()
        
        # Aplicar estilos finais
        self.apply_final_styles()

    def get_modern_stylesheet(self):
        """Retorna um stylesheet moderno para a interface"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }
        
        QPushButton {
            background-color: #4a90e2;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 11pt;
            min-height: 20px;
        }
        
        QPushButton:hover {
            background-color: #357abd;
            transform: translateY(-1px);
        }
        
        QPushButton:pressed {
            background-color: #2d5aa0;
        }
        
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        
        QLabel {
            color: #ffffff;
            font-weight: bold;
            padding: 5px;
            background-color: transparent;
        }
        
        QLabel[class="title"] {
            font-size: 14pt;
            color: #4a90e2;
            background-color: #1e1e1e;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
        }
        
        QLabel[class="analysis"] {
            background-color: #1e1e1e;
            border: 1px solid #4a90e2;
            border-radius: 5px;
            padding: 8px;
            margin: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
        }
        
        QMenuBar {
            background-color: #1e1e1e;
            color: #ffffff;
            border-bottom: 2px solid #4a90e2;
            font-weight: bold;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 12px;
            margin: 2px;
        }
        
        QMenuBar::item:selected {
            background-color: #4a90e2;
            border-radius: 3px;
        }
        
        QMenu {
            background-color: #1e1e1e;
            border: 1px solid #4a90e2;
            border-radius: 5px;
            padding: 5px;
        }
        
        QMenu::item {
            padding: 8px 20px;
            border-radius: 3px;
        }
        
        QMenu::item:selected {
            background-color: #4a90e2;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #4a90e2;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #1e1e1e;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #4a90e2;
        }
        
        QFrame[class="separator"] {
            background-color: #4a90e2;
            border: none;
            min-height: 2px;
            max-height: 2px;
        }
        
        QFrame[class="panel"] {
            background-color: #1e1e1e;
            border: 1px solid #4a90e2;
            border-radius: 8px;
            padding: 10px;
        }
        """

    def setup_color_palette(self):
        """Configura a paleta de cores da aplicação"""
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(74, 144, 226))
        palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(74, 144, 226))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(74, 144, 226))
        palette.setColor(QPalette.Highlight, QColor(74, 144, 226))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)

    def setup_3d_widgets(self):
        """Configura os widgets 3D com melhor aparência"""
        self.gl_original = GLViewWidget()
        self.gl_reparada = GLViewWidget()
        
        # Configurações dos widgets 3D
        for gl_widget in [self.gl_original, self.gl_reparada]:
            gl_widget.setCameraPosition(distance=200)
            gl_widget.setBackgroundColor('w')
            gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            gl_widget.setMinimumSize(400, 300)
            
            # Adiciona borda e estilo aos widgets 3D
            gl_widget.setStyleSheet("""
                QWidget {
                    border: 2px solid #4a90e2;
                    border-radius: 8px;
                    background-color: #ffffff;
                }
            """)

    def setup_labels(self):
        """Configura os labels com melhor estilo"""
        self.label_original = QLabel('🔄 Original')
        self.label_reparada = QLabel('✅ Reparada')
        self.label_analise = QLabel('')
        self.label_analise_reparada = QLabel('')
        
        # Aplicar estilos aos labels
        for label in [self.label_original, self.label_reparada]:
            label.setAlignment(Qt.AlignHCenter)
            label.setProperty("class", "title")
            label.setStyleSheet(label.styleSheet())
        
        for label in [self.label_analise, self.label_analise_reparada]:
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setProperty("class", "analysis")
            label.setStyleSheet(label.styleSheet())
            label.setWordWrap(True)
            label.setMinimumHeight(80)

    def setup_main_buttons(self):
        """Configura os botões principais com design moderno"""
        self.btn_abrir = QPushButton('📁 Abrir STL/OBJ')
        self.btn_reparar = QPushButton('🔧 Reparar Malha')
        self.btn_salvar = QPushButton('💾 Salvar Malha Reparada')
        
        # Adicionar tooltips informativos
        self.btn_abrir.setToolTip('Carregar arquivo STL ou OBJ para reparo')
        self.btn_reparar.setToolTip('Executar reparo automático da malha 3D')
        self.btn_salvar.setToolTip('Salvar malha reparada em formato STL ou OBJ')
        
        # Conectar sinais
        self.btn_abrir.clicked.connect(self.abrir_arquivo)
        self.btn_reparar.clicked.connect(self.reparar_malha)
        self.btn_salvar.clicked.connect(self.salvar_malha)
        
        # Configurar estados iniciais
        self.btn_abrir.setEnabled(True)
        self.btn_reparar.setEnabled(False)
        self.btn_salvar.setEnabled(False)
        
        # Adicionar botões de reset com melhor estilo
        self.btn_reset_original = QPushButton('🔄 Reset Original')
        self.btn_reset_reparada = QPushButton('🔄 Reset Reparada')
        self.btn_reset_original.clicked.connect(lambda: self.centralizar_camera(self.gl_original, self.mesh_original))
        self.btn_reset_reparada.clicked.connect(lambda: self.centralizar_camera(self.gl_reparada, self.mesh_reparada))
        self.btn_reset_original.setEnabled(False)
        self.btn_reset_reparada.setEnabled(False)
        
        # Tooltips para botões de reset
        self.btn_reset_original.setToolTip('Centralizar visualização da malha original')
        self.btn_reset_reparada.setToolTip('Centralizar visualização da malha reparada')

    def setup_main_layout(self, central_widget):
        """Configura o layout principal com melhor organização"""
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Painel superior com botões principais
        top_panel = QFrame()
        top_panel.setProperty("class", "panel")
        top_layout = QVBoxLayout()
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(20, 20, 20, 20)
        
        # Título principal
        title_label = QLabel('🔧 FERRAMENTA PROFISSIONAL DE REPARO DE MALHA 3D')
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #4a90e2;
                background-color: #1e1e1e;
                border: 2px solid #4a90e2;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
            }
        """)
        top_layout.addWidget(title_label)
        
        # Subtítulo informativo
        subtitle_label = QLabel('Reparo automático e profissional de malhas 3D para impressão e modelagem')
        subtitle_label.setAlignment(Qt.AlignHCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                color: #cccccc;
                background-color: transparent;
                padding: 5px;
            }
        """)
        top_layout.addWidget(subtitle_label)
        
        # Botões principais em linha com melhor espaçamento
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_abrir)
        btn_layout.addWidget(self.btn_reparar)
        btn_layout.addWidget(self.btn_salvar)
        btn_layout.addStretch()
        top_layout.addLayout(btn_layout)
        
        top_panel.setLayout(top_layout)
        main_layout.addWidget(top_panel)
        
        # Separador visual
        separator = QFrame()
        separator.setProperty("class", "separator")
        separator.setMaximumHeight(3)
        main_layout.addWidget(separator)
        
        # Layout de visualização com melhor organização
        vis_layout = QHBoxLayout()
        vis_layout.setSpacing(20)
        
        # Painel esquerdo (Original)
        left_panel = QFrame()
        left_panel.setProperty("class", "panel")
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        left_layout.addWidget(self.gl_original)
        left_layout.addWidget(self.label_original)
        left_layout.addWidget(self.label_analise)
        
        # Botões de controle para visualização original
        left_controls = QHBoxLayout()
        left_controls.setSpacing(10)
        left_controls.addWidget(self.btn_reset_original)
        left_controls.addStretch()
        left_layout.addLayout(left_controls)
        
        left_panel.setLayout(left_layout)
        
        # Painel direito (Reparada)
        right_panel = QFrame()
        right_panel.setProperty("class", "panel")
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        right_layout.addWidget(self.gl_reparada)
        right_layout.addWidget(self.label_reparada)
        right_layout.addWidget(self.label_analise_reparada)
        
        # Botões de controle para visualização reparada
        right_controls = QHBoxLayout()
        right_controls.setSpacing(10)
        right_controls.addWidget(self.btn_reset_reparada)
        right_controls.addStretch()
        right_layout.addLayout(right_controls)
        
        right_panel.setLayout(right_layout)
        
        # Adicionar painéis ao layout de visualização
        vis_layout.addWidget(left_panel)
        vis_layout.addWidget(right_panel)
        
        main_layout.addLayout(vis_layout)
        
        # Barra de status
        self.status_bar = QLabel('✅ Pronto para carregar arquivos STL/OBJ')
        self.status_bar.setAlignment(Qt.AlignHCenter)
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #4a90e2;
                border-radius: 5px;
                padding: 8px;
                margin: 5px;
                color: #4a90e2;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(self.status_bar)
        
        # Configurar o layout principal
        central_widget.setLayout(main_layout)

    def setup_menu_bar(self):
        """Configura a barra de menu com estilo melhorado"""
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        
        # Aplicar estilo personalizado ao menu
        self.menu_bar.setStyleSheet(self.menu_bar.styleSheet())

    def setup_actions(self):
        """Configura as ações do menu (será implementado depois)"""
        # Menu Topologia/Geometria
        self.menu_topologia = self.menu_bar.addMenu('🔧 Topologia/Geometria')
        
        self.action_remove_doubles = QAction('🔄 Remover Duplicados', self)
        self.action_remove_doubles.triggered.connect(self.remover_duplicados_dialog)
        self.menu_topologia.addAction(self.action_remove_doubles)
        
        self.action_suavizar = QAction('✨ Suavizar Malha Reparada', self)
        self.action_suavizar.triggered.connect(self.suavizar_malha)
        self.menu_topologia.addAction(self.action_suavizar)
        
        self.action_normals_out = QAction('⬆️ Recalcular Normais para Fora', self)
        self.action_normals_out.triggered.connect(lambda: self.recalcular_normais('out'))
        self.menu_topologia.addAction(self.action_normals_out)
        
        self.action_normals_in = QAction('⬇️ Recalcular Normais para Dentro', self)
        self.action_normals_in.triggered.connect(lambda: self.recalcular_normais('in'))
        self.menu_topologia.addAction(self.action_normals_in)
        
        self.action_fill_holes = QAction('🕳️ Preencher Buracos / Tornar Manifold', self)
        self.action_fill_holes.triggered.connect(self.preencher_buracos)
        self.menu_topologia.addAction(self.action_fill_holes)
        
        self.action_remove_nonmanifold = QAction('❌ Remover Geometria Não-Manifold', self)
        self.action_remove_nonmanifold.triggered.connect(self.remover_nao_manifold)
        self.menu_topologia.addAction(self.action_remove_nonmanifold)
        
        self.action_decimate = QAction('📉 Simplificar Malha (Decimate)', self)
        self.action_decimate.triggered.connect(self.simplificar_malha_dialog)
        self.menu_topologia.addAction(self.action_decimate)
        
        self.action_triangulate = QAction('🔺 Triangular Faces (Triangulate)', self)
        self.action_triangulate.triggered.connect(self.triangulate_faces)
        self.menu_topologia.addAction(self.action_triangulate)
        
        self.action_quadrangulate = QAction('⬜ Quadrangular Faces (Quadrangulate)', self)
        self.action_quadrangulate.triggered.connect(self.quadrangulate_faces)
        self.menu_topologia.addAction(self.action_quadrangulate)
        
        self.action_remove_degenerate = QAction('🗑️ Remover Faces Degeneradas/Área Zero', self)
        self.action_remove_degenerate.triggered.connect(self.remover_faces_degeneradas)
        self.menu_topologia.addAction(self.action_remove_degenerate)
        
        self.action_remesh = QAction('🔲 Remesh (Voxel)', self)
        self.action_remesh.triggered.connect(self.remesh_voxel_dialog)
        self.menu_topologia.addAction(self.action_remesh)
        
        self.action_remesh_surface = QAction('🌊 Remesh (Surface)', self)
        self.action_remesh_surface.triggered.connect(self.remesh_surface_dialog)
        self.menu_topologia.addAction(self.action_remesh_surface)
        
        self.action_auto_retopo = QAction('🔄 Auto Retopology', self)
        self.action_auto_retopo.triggered.connect(self.auto_retopology_dialog)
        self.menu_topologia.addAction(self.action_auto_retopo)
        
        # Menu Sombreamento
        self.menu_sombreamento = self.menu_bar.addMenu('🎨 Sombreamento')
        
        self.action_shade_smooth = QAction('✨ Shade Smooth', self)
        self.action_shade_smooth.triggered.connect(self.shade_smooth)
        self.menu_sombreamento.addAction(self.action_shade_smooth)
        
        self.action_shade_flat = QAction('📐 Shade Flat', self)
        self.action_shade_flat.triggered.connect(self.shade_flat)
        self.menu_sombreamento.addAction(self.action_shade_flat)
        
        self.action_auto_smooth = QAction('🤖 Auto Smooth', self)
        self.action_auto_smooth.triggered.connect(self.auto_smooth_dialog)
        self.menu_sombreamento.addAction(self.action_auto_smooth)
        
        self.action_transfer_normals = QAction('📤 Transferir Normais da Original', self)
        self.action_transfer_normals.triggered.connect(self.transferir_normais)
        self.menu_sombreamento.addAction(self.action_transfer_normals)
        
        self.action_weighted_normals = QAction('⚖️ Weighted Normals', self)
        self.action_weighted_normals.triggered.connect(self.weighted_normals)
        self.menu_sombreamento.addAction(self.action_weighted_normals)
        
        self.action_split_normals = QAction('✂️ Split Normals (Hard Edges)', self)
        self.action_split_normals.triggered.connect(self.split_normals_dialog)
        self.menu_sombreamento.addAction(self.action_split_normals)
        
        # Menu Malha e Estrutura
        self.menu_malha = self.menu_bar.addMenu('🏗️ Malha e Estrutura')
        
        self.action_mesh_cleanup = QAction('🧹 Mesh Cleanup / Delete Loose Geometry', self)
        self.action_mesh_cleanup.triggered.connect(self.mesh_cleanup)
        self.menu_malha.addAction(self.action_mesh_cleanup)
        
        self.action_edge_split = QAction('✂️ Edge Split Modifier', self)
        self.action_edge_split.triggered.connect(self.edge_split_dialog)
        self.menu_malha.addAction(self.action_edge_split)
        
        self.action_remove_interior = QAction('🚫 Remover Faces Interiores/Intersectantes', self)
        self.action_remove_interior.triggered.connect(self.remover_faces_interiores)
        self.menu_malha.addAction(self.action_remove_interior)
        
        self.action_weld_vertices = QAction('🔗 Soldar Vértices / Colapsar Arestas', self)
        self.action_weld_vertices.triggered.connect(self.weld_vertices_dialog)
        self.menu_malha.addAction(self.action_weld_vertices)
        
        self.action_subdivision = QAction('🔲 Subdivision Surface / Catmull-Clark', self)
        self.action_subdivision.triggered.connect(self.subdivision_surface_dialog)
        self.menu_malha.addAction(self.action_subdivision)
        
        self.action_solidify = QAction('🧱 Solidify Modifier (Casca Espessa)', self)
        self.action_solidify.triggered.connect(self.solidify_modifier_dialog)
        self.menu_malha.addAction(self.action_solidify)
        
        # Menu Verificações e Diagnóstico
        self.menu_verificacoes = self.menu_bar.addMenu('🔍 Verificações e Diagnóstico')
        
        self.action_mesh_stats = QAction('📊 Estatísticas da Malha (N-gons, Tris, Pólos)', self)
        self.action_mesh_stats.triggered.connect(self.estatisticas_malha_dialog)
        self.menu_verificacoes.addAction(self.action_mesh_stats)
        
        # Menu Visualização
        self.menu_visualizacao = self.menu_bar.addMenu('👁️ Visualização')
        
        self.action_reset_original = QAction('🔄 Resetar Visualização Original', self)
        self.action_reset_original.triggered.connect(lambda: self.centralizar_camera(self.gl_original, self.mesh_original))
        self.menu_visualizacao.addAction(self.action_reset_original)
        
        self.action_reset_reparada = QAction('🔄 Resetar Visualização Reparada', self)
        self.action_reset_reparada.triggered.connect(lambda: self.centralizar_camera(self.gl_reparada, self.mesh_reparada))
        self.menu_visualizacao.addAction(self.action_reset_reparada)
        
        # Menu Ferramentas (para métodos PyMeshLab)
        self.menu_ferramentas = self.menu_bar.addMenu('🛠️ Ferramentas')
        
        self.action_listar_metodos = QAction('📋 Listar Métodos PyMeshLab', self)
        self.action_listar_metodos.triggered.connect(self.listar_metodos_pymeshlab)
        self.menu_ferramentas.addAction(self.action_listar_metodos)
        
        self.action_exportar_metodos = QAction('💾 Exportar Métodos PyMeshLab', self)
        self.action_exportar_metodos.triggered.connect(self.exportar_metodos_pymeshlab)
        self.menu_ferramentas.addAction(self.action_exportar_metodos)
        
        self.action_exportar_atributos = QAction('📄 Exportar Atributos PyMeshLab', self)
        self.action_exportar_atributos.triggered.connect(self.exportar_atributos_pymeshlab)
        self.menu_ferramentas.addAction(self.action_exportar_atributos)
        
        # Habilitar/desabilitar ações conforme contexto
        self.disable_all_actions()
        
        # Ações que sempre estão habilitadas
        self.action_reset_original.setEnabled(True)
        self.action_listar_metodos.setEnabled(True)
        self.action_exportar_metodos.setEnabled(True)
        self.action_exportar_atributos.setEnabled(True)

    def disable_all_actions(self):
        """Desabilita todas as ações que dependem de malha carregada"""
        actions_to_disable = [
            self.action_remove_doubles, self.action_suavizar, self.action_normals_out,
            self.action_normals_in, self.action_fill_holes, self.action_remove_nonmanifold,
            self.action_decimate, self.action_triangulate, self.action_quadrangulate,
            self.action_remove_degenerate, self.action_remesh, self.action_remesh_surface,
            self.action_auto_retopo, self.action_shade_smooth, self.action_shade_flat,
            self.action_auto_smooth, self.action_transfer_normals, self.action_weighted_normals,
            self.action_split_normals, self.action_mesh_cleanup, self.action_edge_split,
            self.action_remove_interior, self.action_weld_vertices, self.action_subdivision,
            self.action_solidify, self.action_mesh_stats, self.action_reset_reparada
        ]
        
        for action in actions_to_disable:
            if action:
                action.setEnabled(False)

    def enable_all_actions(self):
        """Habilita todas as ações quando malha é reparada"""
        actions_to_enable = [
            self.action_remove_doubles, self.action_suavizar, self.action_normals_out,
            self.action_normals_in, self.action_fill_holes, self.action_remove_nonmanifold,
            self.action_decimate, self.action_triangulate, self.action_quadrangulate,
            self.action_remove_degenerate, self.action_remesh, self.action_remesh_surface,
            self.action_auto_retopo, self.action_shade_smooth, self.action_shade_flat,
            self.action_auto_smooth, self.action_transfer_normals, self.action_weighted_normals,
            self.action_split_normals, self.action_mesh_cleanup, self.action_edge_split,
            self.action_remove_interior, self.action_weld_vertices, self.action_subdivision,
            self.action_solidify, self.action_mesh_stats, self.action_reset_reparada
        ]
        
        for action in actions_to_enable:
            if action:
                action.setEnabled(True)

    def setup_timers(self):
        """Configura timers para atualizações visuais"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visual_elements)
        self.update_timer.start(100)  # Atualiza a cada 100ms

    def apply_final_styles(self):
        """Aplica estilos finais e configurações visuais"""
        # Forçar atualização dos estilos
        self.setStyleSheet(self.styleSheet())
        
        # Configurar fonte padrão
        font = QFont("Segoe UI", 10)
        self.setFont(font)

    def update_visual_elements(self):
        """Atualiza elementos visuais periodicamente"""
        # Atualizar estados dos botões baseado na disponibilidade das malhas
        if hasattr(self, 'btn_reset_original'):
            self.btn_reset_original.setEnabled(self.mesh_original is not None)
        if hasattr(self, 'btn_reset_reparada'):
            self.btn_reset_reparada.setEnabled(self.mesh_reparada is not None)

    def update_status_bar(self, message, status_type="info"):
        """Atualiza a barra de status com mensagem e tipo específico"""
        if not hasattr(self, 'status_bar'):
            return
            
        # Definir cores baseadas no tipo de status
        colors = {
            "info": "#4a90e2",      # Azul para informações
            "success": "#4CAF50",   # Verde para sucesso
            "warning": "#FF9800",   # Laranja para avisos
            "error": "#F44336"      # Vermelho para erros
        }
        
        color = colors.get(status_type, colors["info"])
        
        # Atualizar estilo e texto
        self.status_bar.setText(message)
        self.status_bar.setStyleSheet(f"""
            QLabel {{
                background-color: #1e1e1e;
                border: 1px solid {color};
                border-radius: 5px;
                padding: 8px;
                margin: 5px;
                color: {color};
                font-weight: bold;
            }}
        """)

    def abrir_arquivo(self):
        from PyQt5.QtWidgets import QMessageBox
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir arquivo STL/OBJ', '', 'Malhas 3D (*.stl *.obj)')
        if fname:
            self.update_status_bar('🔄 Carregando arquivo...', 'info')
            mesh = trimesh.load(fname, process=False)
            # Tentar processar e corrigir problemas leves
            try:
                mesh.process(validate=True)
            except Exception as e:
                QMessageBox.warning(self, 'Aviso', f'Problemas ao processar a malha original.\n{e}')
                self.update_status_bar('⚠️ Problemas detectados na malha original', 'warning')
            mesh = centralizar_na_origem(mesh)
            self.mesh_original = mesh
            self.gl_original.clear()
            self.gl_reparada.clear()
            try:
                item = create_glmeshitem(self.mesh_original, color=(0.1, 0.3, 1, 1))  # azul mais forte
                self.gl_original.addItem(item)
                self.update_status_bar(f'✅ Arquivo carregado: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces', 'success')
            except Exception as e:
                QMessageBox.warning(self, 'Erro ao Renderizar', f'Não foi possível renderizar a malha original.\nA malha pode estar corrompida ou precisar de reparo.\n{e}')
                self.update_status_bar('❌ Erro ao renderizar malha', 'error')
            self.highlight_holes(self.mesh_original)
            self.highlight_nonmanifold_faces(self.mesh_original)
            self.analisar_malha(self.mesh_original, self.label_analise)
            self.label_analise_reparada.setText('')
            self.mesh_reparada = None
            self.btn_reparar.setEnabled(True)
            self.btn_salvar.setEnabled(False)
            self.disable_all_actions()
            self.action_reset_original.setEnabled(True)
            self.centralizar_camera(self.gl_original, self.mesh_original)
            self.centralizar_camera(self.gl_reparada, self.mesh_original)

    def highlight_holes(self, mesh):
        # Encontra arestas abertas (buracos)
        if not hasattr(mesh, 'edges_open'):
            return
        open_edges = mesh.edges_open
        if open_edges.shape[0] == 0:
            return  # Não há buracos
        for edge in open_edges:
            v0, v1 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
            pts = np.array([v0, v1])
            line = GLLinePlotItem(pos=pts, color=(1,0,0,1), width=6, antialias=True, mode='lines')
            self.gl_original.addItem(line)

    def highlight_nonmanifold_faces(self, mesh):
        # Destaca faces não-manifold em laranja
        if hasattr(mesh, 'faces_nonmanifold') and len(mesh.faces_nonmanifold) > 0:
            faces_idx = mesh.faces_nonmanifold
            vertices = mesh.vertices
            for face in faces_idx:
                pts = vertices[mesh.faces[face]]
                # Fechar o triângulo para desenhar
                pts = np.vstack([pts, pts[0]])
                line = GLLinePlotItem(pos=pts, color=(1,0.5,0,1), width=4, antialias=True, mode='line_strip')
                self.gl_original.addItem(line)

    def analisar_malha(self, mesh, label):
        n_vertices = len(mesh.vertices)
        n_faces = len(mesh.faces)
        n_buracos = len(mesh.edges_open) if hasattr(mesh, 'edges_open') else 0
        watertight = mesh.is_watertight
        n_nonmanifold = len(mesh.faces_nonmanifold) if hasattr(mesh, 'faces_nonmanifold') else 0
        n_duplicados = len(mesh.vertices) - len(np.unique(mesh.vertices, axis=0))
        texto = f"<b>Análise da Malha:</b><br>"
        texto += f"Vértices: {n_vertices}<br>"
        texto += f"Faces: {n_faces}<br>"
        texto += f"Buracos (arestas abertas): {n_buracos}<br>"
        texto += f"Watertight: {'Sim' if watertight else 'Não'}<br>"
        texto += f"Faces não-manifold: {n_nonmanifold}<br>"
        texto += f"Vértices duplicados: {n_duplicados}<br>"
        label.setText(texto)

    def reparar_malha(self):
        if self.mesh_original is None:
            return
        try:
            self.update_status_bar('🔧 Reparando malha...', 'info')
            mesh = self.mesh_original
            if mesh.is_watertight:
                mesh_reparada = mesh.copy()
                self.update_status_bar('ℹ️ Malha já está fechada (watertight)', 'info')
            else:
                meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                meshfix.repair(verbose=True)
                mesh_reparada = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
                self.update_status_bar('🔧 Aplicando reparos automáticos...', 'info')
            
            mesh_reparada = centralizar_na_origem(mesh_reparada)
            self.mesh_reparada = mesh_reparada
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh_reparada, color=(0.1, 0.8, 0.1, 1))  # verde mais forte
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh_reparada, self.label_analise_reparada)
            self.btn_salvar.setEnabled(True)
            
            # Habilitar todas as ações quando malha é reparada
            self.enable_all_actions()
            self.centralizar_camera(self.gl_reparada, mesh_reparada)
            
            self.update_status_bar(f'✅ Malha reparada com sucesso: {len(mesh_reparada.vertices)} vértices, {len(mesh_reparada.faces)} faces', 'success')
            
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Reparar', f'Não foi possível reparar a malha.\n{e}')
            self.update_status_bar('❌ Erro ao reparar malha', 'error')

    def salvar_malha(self):
        if self.mesh_reparada is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, 'Salvar Malha Reparada', '', 'Malhas 3D (*.stl *.obj)')
        if fname:
            self.mesh_reparada.export(fname)

    def centralizar_camera(self, gl_widget, mesh):
        # Centraliza e ajusta o zoom da câmera para enquadrar a peça
        if mesh is None or not hasattr(mesh, 'bounding_box'):
            return
        bbox = mesh.bounding_box
        ext = bbox.extents
        dist = max(ext) * 2.5 if max(ext) > 0 else 100
        gl_widget.setCameraPosition(distance=dist)

    def suavizar_malha(self):
        if self.mesh_reparada is None:
            return
        # Suavização simples: laplacian smoothing
        try:
            smoothed = self.mesh_reparada.copy()
            smoothed = trimesh.graph.smooth_shade(smoothed)
            self.mesh_reparada = smoothed
            self.gl_reparada.clear()
            item = create_glmeshitem(smoothed, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(smoothed, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, smoothed)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Suavizar', f'Não foi possível suavizar a malha.\n{e}')

    def remover_duplicados_dialog(self):
        if self.mesh_reparada is None:
            return
        threshold, ok = QInputDialog.getDouble(self, 'Remover Duplicados', 'Distância de mesclagem:', 0.0001, 0.0, 1.0, 6)
        if ok:
            self.remover_duplicados(threshold)

    def remover_duplicados(self, threshold):
        if self.mesh_reparada is None:
            return
        mesh = self.mesh_reparada.copy()
        try:
            # Remover vértices duplicados manualmente
            verts = mesh.vertices
            faces = mesh.faces
            tree = cKDTree(verts)
            groups = tree.query_ball_point(verts, threshold)
            # Mapear cada vértice para o menor índice do seu grupo
            mapping = np.arange(len(verts))
            for i, group in enumerate(groups):
                min_idx = min(group)
                mapping[i] = min_idx
            # Atualizar faces
            new_faces = mapping[faces]
            # Remover vértices não usados
            unique, inverse = np.unique(new_faces, return_inverse=True)
            new_verts = verts[unique]
            new_faces = inverse.reshape(new_faces.shape)
            merged = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
            self.mesh_reparada = merged
            self.gl_reparada.clear()
            item = create_glmeshitem(merged, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(merged, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, merged)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Remover Duplicados', f'Não foi possível remover duplicados.\n{e}')

    def recalcular_normais(self, orientacao):
        if self.mesh_reparada is None:
            return
        mesh = self.mesh_reparada.copy()
        try:
            if orientacao == 'out':
                mesh.rezero()
                mesh.fix_normals()
            elif orientacao == 'in':
                mesh.rezero()
                mesh.fix_normals()
                mesh.invert()
            self.mesh_reparada = mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, mesh)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Recalcular Normais', f'Não foi possível recalcular as normais.\n{e}')

    def preencher_buracos(self):
        if self.mesh_reparada is None:
            return
        mesh = self.mesh_reparada.copy()
        try:
            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(verbose=True)
            manifold = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
            self.mesh_reparada = manifold
            self.gl_reparada.clear()
            item = create_glmeshitem(manifold, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(manifold, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, manifold)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Preencher Buracos', f'Não foi possível preencher buracos/tornar manifold.\n{e}')

    def remover_nao_manifold(self):
        if self.mesh_reparada is None:
            return
        mesh = self.mesh_reparada.copy()
        try:
            # Remove faces não-manifold
            if hasattr(mesh, 'faces_nonmanifold') and len(mesh.faces_nonmanifold) > 0:
                mask = np.ones(len(mesh.faces), dtype=bool)
                mask[mesh.faces_nonmanifold] = False
                new_faces = mesh.faces[mask]
                cleaned = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces, process=True)
            else:
                cleaned = mesh
            self.mesh_reparada = cleaned
            self.gl_reparada.clear()
            item = create_glmeshitem(cleaned, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(cleaned, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, cleaned)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Remover Não-Manifold', f'Não foi possível remover geometria não-manifold.\n{e}')

    def simplificar_malha_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog
        fator, ok = QInputDialog.getDouble(self, 'Simplificar Malha', 'Fator de redução (0.0 a 1.0):', 0.5, 0.01, 1.0, 2)
        if ok:
            self.simplificar_malha(fator)

    def simplificar_malha(self, fator):
        if self.mesh_reparada is None:
            return
        import pymeshlab
        import numpy as np
        try:
            m = self.mesh_reparada
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(len(m.faces)*fator), preservenormal=True)
            new_mesh = ms.current_mesh()
            vertices = np.array(new_mesh.vertex_matrix())
            faces = np.array(new_mesh.face_matrix())
            simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            simplified = centralizar_na_origem(simplified)
            self.mesh_reparada = simplified
            self.gl_reparada.clear()
            item = create_glmeshitem(simplified, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(simplified, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, simplified)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Erro ao Simplificar', f'Não foi possível simplificar a malha.\n{e}')

    def triangulate_faces(self):
        if self.mesh_reparada is None:
            return
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        faces = self.mesh_reparada.faces
        # Detectar se já está toda em triângulos
        if np.all([len(set(face)) == 3 for face in faces]):
            QMessageBox.information(self, 'Triangular Faces', 'A malha já está toda em triângulos!')
            return
        try:
            verts = self.mesh_reparada.vertices
            new_faces = []
            for face in faces:
                unique = list(dict.fromkeys(face))
                if len(unique) < 3:
                    continue
                if len(unique) == 3:
                    new_faces.append(unique)
                else:
                    # Fan triangulation
                    for i in range(1, len(unique) - 1):
                        new_faces.append([unique[0], unique[i], unique[i+1]])
            tri = trimesh.Trimesh(vertices=verts, faces=np.array(new_faces), process=True)
            tri = centralizar_na_origem(tri)
            self.mesh_reparada = tri
            self.gl_reparada.clear()
            item = create_glmeshitem(tri, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(tri, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, tri)
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao Triangular', f'Não foi possível triangular as faces.\n{e}')

    def quadrangulate_faces(self):
        if self.mesh_reparada is None:
            return
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        faces = self.mesh_reparada.faces
        verts = self.mesh_reparada.vertices
        quads = []
        # Agrupar pares de triângulos adjacentes em quadriláteros
        # (Simples: só para faces já trianguladas)
        if np.all([len(set(face)) == 3 for face in faces]):
            from collections import defaultdict
            edge_map = defaultdict(list)
            for idx, face in enumerate(faces):
                for i in range(3):
                    a, b = sorted((face[i], face[(i+1)%3]))
                    edge_map[(a, b)].append(idx)
            paired = set()
            for idx, face in enumerate(faces):
                if idx in paired:
                    continue
                found = False
                for i in range(3):
                    a, b = sorted((face[i], face[(i+1)%3]))
                    adj = [f for f in edge_map[(a, b)] if f != idx and f not in paired]
                    if adj:
                        j = adj[0]
                        f2 = faces[j]
                        quad = list(set(face) | set(f2))
                        if len(quad) == 4:
                            quads.append(quad)
                            paired.add(idx)
                            paired.add(j)
                            found = True
                            break
            # Só criar malha se todas as faces foram agrupadas em quadriláteros
            if len(quads)*2 == len(faces):
                try:
                    quad = trimesh.Trimesh(vertices=verts, faces=np.array(quads), process=True)
                    quad = centralizar_na_origem(quad)
                    self.mesh_reparada = quad
                    self.gl_reparada.clear()
                    item = create_glmeshitem(quad, color=(0.1, 0.8, 0.1, 1))
                    self.gl_reparada.addItem(item)
                    self.analisar_malha(quad, self.label_analise_reparada)
                    self.centralizar_camera(self.gl_reparada, quad)
                except Exception as e:
                    QMessageBox.warning(self, 'Erro ao Quadrangular', f'Não foi possível quadrangular as faces.\n{e}')
            else:
                QMessageBox.warning(self, 'Quadrangular Faces', 'Não foi possível quadrangular toda a malha. Só é possível quadrangular se todos os triângulos puderem ser agrupados em pares.')
        else:
            QMessageBox.warning(self, 'Quadrangular Faces', 'A quadrangulação automática só é suportada para malhas totalmente trianguladas.')

    def remover_faces_degeneradas(self):
        if self.mesh_reparada is None:
            return
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        mesh = self.mesh_reparada.copy()
        # Faces degeneradas: área zero ou vértices repetidos/colineares
        areas = mesh.area_faces
        mask = areas > 1e-12
        faces_validas = mesh.faces[mask]
        try:
            cleaned = trimesh.Trimesh(vertices=mesh.vertices, faces=faces_validas, process=True)
            cleaned = centralizar_na_origem(cleaned)
            self.mesh_reparada = cleaned
            self.gl_reparada.clear()
            item = create_glmeshitem(cleaned, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(cleaned, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, cleaned)
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao Remover Faces Degeneradas', f'Não foi possível remover faces degeneradas.\n{e}')

    def remesh_voxel_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog
        tamanho, ok = QInputDialog.getDouble(self, 'Remesh (Voxel)', 'Tamanho do voxel:', 1.0, 0.001, 100.0, 3)
        if ok:
            self.remesh_voxel(tamanho)

    def remesh_voxel(self, voxel_size):
        if self.mesh_reparada is None:
            return
        import trimesh
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            # Voxel remesh
            v = mesh.voxelized(voxel_size)
            remeshed = v.as_boxes()
            remeshed = centralizar_na_origem(remeshed)
            self.mesh_reparada = remeshed
            self.gl_reparada.clear()
            item = create_glmeshitem(remeshed, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(remeshed, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, remeshed)
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao Remesh', f'Não foi possível refazer a malha (remesh).\n{e}')

    def remesh_surface_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        length, ok = QInputDialog.getDouble(self, 'Remesh (Surface)', 'Comprimento alvo da aresta:', 1.0, 0.001, 100.0, 3)
        if ok:
            if length <= 0:
                QMessageBox.warning(self, 'Valor inválido', 'O comprimento da aresta deve ser positivo.')
                return
            self.remesh_surface(length)

    def remesh_surface(self, edge_length):
        if self.mesh_reparada is None:
            return
        import pymeshlab
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            m = self.mesh_reparada
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
            ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_length))
            new_mesh = ms.current_mesh()
            vertices = np.array(new_mesh.vertex_matrix())
            faces = np.array(new_mesh.face_matrix())
            remeshed = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            remeshed = centralizar_na_origem(remeshed)
            self.mesh_reparada = remeshed
            self.gl_reparada.clear()
            item = create_glmeshitem(remeshed, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(remeshed, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, remeshed)
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao Remesh (Surface)', f'Não foi possível refazer a malha por superfície.\n{e}')

    def listar_metodos_pymeshlab(self):
        import pymeshlab
        from PyQt5.QtWidgets import QMessageBox
        metodos = dir(pymeshlab.MeshSet)
        metodos_str = '\n'.join([m for m in metodos if not m.startswith('_')])
        QMessageBox.information(self, 'Métodos do PyMeshLab.MeshSet', metodos_str)

    def exportar_metodos_pymeshlab(self):
        import pymeshlab
        metodos = dir(pymeshlab.MeshSet)
        metodos_str = '\n'.join([m for m in metodos if not m.startswith('_')])
        with open('metodos_pymeshlab.txt', 'w', encoding='utf-8') as f:
            f.write(metodos_str)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, 'Exportação concluída', 'Lista de métodos salva em metodos_pymeshlab.txt')

    def exportar_atributos_pymeshlab(self):
        import pymeshlab
        atributos = dir(pymeshlab)
        atributos_str = '\n'.join([a for a in atributos if not a.startswith('_')])
        with open('atributos_pymeshlab.txt', 'w', encoding='utf-8') as f:
            f.write(atributos_str)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, 'Exportação concluída', 'Lista de atributos salva em atributos_pymeshlab.txt')

    def auto_retopology_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog
        faces, ok1 = QInputDialog.getInt(self, 'Auto Retopology', 'Número alvo de faces (simplificação):', 1000, 100, 100000, 1)
        if not ok1:
            return
        edge, ok2 = QInputDialog.getDouble(self, 'Auto Retopology', 'Comprimento alvo da aresta (remesh):', 1.0, 0.001, 100.0, 3)
        if not ok2 or edge <= 0:
            return
        self.auto_retopology(faces, edge)

    def auto_retopology(self, target_faces, edge_length):
        if self.mesh_reparada is None:
            return
        import pymeshlab
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            # 1. Simplificação
            m = self.mesh_reparada
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True)
            new_mesh = ms.current_mesh()
            vertices = np.array(new_mesh.vertex_matrix())
            faces = np.array(new_mesh.face_matrix())
            simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            # 2. Remesh Surface
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(simplified.vertices, simplified.faces))
            ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_length))
            new_mesh = ms.current_mesh()
            vertices = np.array(new_mesh.vertex_matrix())
            faces = np.array(new_mesh.face_matrix())
            remeshed = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            # 3. Suavização
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(remeshed.vertices, remeshed.faces))
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=10)
            new_mesh = ms.current_mesh()
            vertices = np.array(new_mesh.vertex_matrix())
            faces = np.array(new_mesh.face_matrix())
            final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            final_mesh = centralizar_na_origem(final_mesh)
            self.mesh_reparada = final_mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(final_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(final_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, final_mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro no Auto Retopology', f'Não foi possível executar auto retopologia.\n{e}')

    def shade_smooth(self):
        if self.mesh_reparada is None:
            return
        import trimesh
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
            self.mesh_reparada = mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro no Shade Smooth', f'Não foi possível aplicar Shade Smooth.\n{e}')

    def shade_flat(self):
        if self.mesh_reparada is None:
            return
        import trimesh
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
            mesh.vertex_normals = None
            self.mesh_reparada = mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro no Shade Flat', f'Não foi possível aplicar Shade Flat.\n{e}')

    def auto_smooth_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog
        angulo, ok = QInputDialog.getDouble(self, 'Auto Smooth', 'Ângulo limite (graus):', 30.0, 1.0, 180.0, 1)
        if ok:
            self.auto_smooth(angulo)

    def auto_smooth(self, angle_limit):
        if self.mesh_reparada is None:
            return
        import trimesh
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            faces = mesh.faces
            verts = mesh.vertices
            face_normals = mesh.face_normals
            # Usa face_adjacency e face_adjacency_edges
            sharp_edges = set()
            rad_limit = np.deg2rad(angle_limit)
            for (f1, f2), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
                n1 = face_normals[f1]
                n2 = face_normals[f2]
                angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
                if angle > rad_limit:
                    sharp_edges.add(tuple(sorted(edge)))
            # Para cada vértice, se está em uma aresta "viva", usa normal da face, senão normal suavizada
            vertex_normals = np.zeros_like(verts)
            counts = np.zeros(len(verts))
            for i, face in enumerate(faces):
                is_sharp = False
                for j in range(len(face)):
                    a = face[j]
                    b = face[(j+1)%len(face)]
                    if tuple(sorted((a,b))) in sharp_edges:
                        is_sharp = True
                        break
                if is_sharp:
                    for idx in face:
                        vertex_normals[idx] += face_normals[i]
                        counts[idx] += 1
                else:
                    for idx in face:
                        vertex_normals[idx] += face_normals[i]
                        counts[idx] += 1
            # Normaliza
            mask = counts > 0
            vertex_normals[mask] /= counts[mask][:,None]
            norms = np.linalg.norm(vertex_normals, axis=1)
            mask = norms > 0
            vertex_normals[mask] /= norms[mask][:,None]
            mesh.vertex_normals = vertex_normals
            self.mesh_reparada = mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro no Auto Smooth', f'Não foi possível aplicar Auto Smooth.\n{e}')

    def transferir_normais(self):
        if self.mesh_original is None or self.mesh_reparada is None:
            return
        import trimesh
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            orig = self.mesh_original
            rep = self.mesh_reparada
            if len(orig.vertices) == len(rep.vertices):
                rep.vertex_normals = orig.vertex_normals.copy()
            else:
                # Transferência por proximidade (k-d tree)
                from scipy.spatial import cKDTree
                tree = cKDTree(orig.vertices)
                dists, idxs = tree.query(rep.vertices)
                rep.vertex_normals = orig.vertex_normals[idxs]
            self.mesh_reparada = rep
            self.gl_reparada.clear()
            item = create_glmeshitem(rep, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(rep, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, rep)
        except Exception as e:
            QMessageBox.warning(self, 'Erro ao Transferir Normais', f'Não foi possível transferir as normais.\n{e}')

    def weighted_normals(self):
        if self.mesh_reparada is None:
            return
        import trimesh
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            verts = mesh.vertices
            faces = mesh.faces
            face_normals = mesh.face_normals
            face_areas = mesh.area_faces
            vertex_normals = np.zeros_like(verts)
            weights = np.zeros(len(verts))
            for i, face in enumerate(faces):
                area = face_areas[i]
                for idx in face:
                    vertex_normals[idx] += face_normals[i] * area
                    weights[idx] += area
            mask = weights > 0
            vertex_normals[mask] /= weights[mask][:,None]
            norms = np.linalg.norm(vertex_normals, axis=1)
            mask = norms > 0
            vertex_normals[mask] /= norms[mask][:,None]
            mesh.vertex_normals = vertex_normals
            self.mesh_reparada = mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro em Weighted Normals', f'Não foi possível aplicar Weighted Normals.\n{e}')

    def split_normals_dialog(self):
        if self.mesh_reparada is None:
            return
        from PyQt5.QtWidgets import QInputDialog
        angulo, ok = QInputDialog.getDouble(self, 'Split Normals (Hard Edges)', 'Ângulo limite (graus):', 30.0, 1.0, 180.0, 1)
        if ok:
            self.split_normals(angulo)

    def split_normals(self, angle_limit):
        if self.mesh_reparada is None:
            return
        import trimesh
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        try:
            mesh = self.mesh_reparada.copy()
            verts = mesh.vertices
            faces = mesh.faces
            face_normals = mesh.face_normals
            # Identifica arestas vivas
            rad_limit = np.deg2rad(angle_limit)
            hard_edges = set()
            for (f1, f2), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
                n1 = face_normals[f1]
                n2 = face_normals[f2]
                angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
                if angle > rad_limit:
                    hard_edges.add(tuple(sorted(edge)))
            # Duplicar vértices nas arestas vivas
            new_verts = verts.tolist()
            new_faces = []
            vert_map = {}  # (orig_idx, face_idx) -> new_idx
            for i, face in enumerate(faces):
                new_face = []
                for j, idx in enumerate(face):
                    prev = face[j-1]
                    edge = tuple(sorted((idx, prev)))
                    if edge in hard_edges:
                        # Duplicar vértice para esta face
                        key = (idx, i)
                        if key not in vert_map:
                            new_verts.append(verts[idx].tolist())
                            vert_map[key] = len(new_verts) - 1
                        new_face.append(vert_map[key])
                    else:
                        new_face.append(idx)
                new_faces.append(new_face)
            new_verts = np.array(new_verts)
            new_faces = np.array(new_faces)
            split_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
            self.mesh_reparada = split_mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(split_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(split_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, split_mesh)
        except Exception as e:
            QMessageBox.warning(self, 'Erro em Split Normals', f'Não foi possível aplicar Split Normals.\n{e}')

    def mesh_cleanup(self):
        if self.mesh_reparada is None:
            return
        import trimesh
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox, QInputDialog
        try:
            mesh = self.mesh_reparada.copy()
            # Remover componentes desconectados pequenos
            min_faces, ok = QInputDialog.getInt(self, 'Mesh Cleanup', 'Mínimo de faces por componente para manter:', 50, 1, 10000, 1)
            if not ok:
                return
            comps = mesh.split(only_watertight=False)
            comps = [c for c in comps if len(c.faces) >= min_faces]
            if not comps:
                QMessageBox.warning(self, 'Mesh Cleanup', 'Nenhuma componente atende ao critério. Nada foi removido.')
                return
            cleaned = trimesh.util.concatenate(comps)
            # Remover vértices soltos
            mask = np.unique(cleaned.faces)
            cleaned.vertices = cleaned.vertices[mask]
            remap = {old: new for new, old in enumerate(mask)}
            cleaned.faces = np.vectorize(remap.get)(cleaned.faces)
            cleaned = trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=True)
            self.mesh_reparada = cleaned
            self.gl_reparada.clear()
            item = create_glmeshitem(cleaned, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(cleaned, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, cleaned)
        except Exception as e:
            QMessageBox.warning(self, 'Erro em Mesh Cleanup', f'Não foi possível limpar a malha.\n{e}')

    def weld_vertices_dialog(self):
        """Diálogo para configurar a soldagem de vértices"""
        if self.mesh_reparada is None:
            return
        
        try:
            threshold, ok = QInputDialog.getDouble(
                self, 'Soldar Vértices', 
                'Distância máxima para soldar vértices:',
                value=0.001, min=0.0001, max=1.0, decimals=6
            )
            
            if ok:
                self.weld_vertices(threshold)
                
        except Exception as e:
            print(f"Erro no diálogo de soldagem: {e}")
            import traceback
            traceback.print_exc()

    def weld_vertices(self, threshold):
        """Solda vértices próximos e colapsa arestas correspondentes"""
        if self.mesh_reparada is None:
            return
        
        try:
            print(f"Soldando vértices com distância máxima: {threshold}")
            
            mesh = self.mesh_reparada.copy()
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Usa KDTree para encontrar vértices próximos
            tree = cKDTree(vertices)
            
            # Encontra grupos de vértices próximos
            groups = tree.query_ball_point(vertices, threshold)
            
            # Cria mapeamento de vértices para o representante do grupo
            vertex_mapping = np.arange(len(vertices))
            for i, group in enumerate(groups):
                # O representante é o vértice com menor índice no grupo
                min_idx = min(group)
                vertex_mapping[i] = min_idx
            
            # Atualiza as faces com os novos índices
            new_faces = vertex_mapping[faces]
            
            # Remove vértices não utilizados
            used_vertices, inverse = np.unique(new_faces, return_inverse=True)
            new_vertices = vertices[used_vertices]
            new_faces = inverse.reshape(new_faces.shape)
            
            # Cria nova malha
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
            
            # Atualiza a malha reparada
            self.mesh_reparada = new_mesh
            
            # Atualiza visualização
            self.gl_reparada.clear()
            item = create_glmeshitem(new_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(new_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, new_mesh)
            
            print(f"Vértices soldados com sucesso. "
                  f"Vértices: {len(vertices)} → {len(new_vertices)}, "
                  f"Faces: {len(faces)} → {len(new_faces)}")
            
        except Exception as e:
            print(f"Erro ao soldar vértices: {e}")
            import traceback
            traceback.print_exc()

    def remover_faces_interiores(self):
        """Remove faces interiores e faces que se intersectam"""
        if self.mesh_reparada is None:
            return
        
        try:
            mesh = self.mesh_reparada.copy()
            
            # Remove faces interiores (faces que não estão na superfície)
            # Uma face é considerada interior se todas suas arestas são compartilhadas por outras faces
            faces_to_remove = set()
            
            # Para cada face, verifica se é interior
            for face_idx, face in enumerate(mesh.faces):
                is_interior = True
                
                # Para cada aresta da face
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    edge_face_count = 0
                    
                    # Conta quantas faces compartilham esta aresta
                    for other_face_idx, other_face in enumerate(mesh.faces):
                        for j in range(3):
                            other_edge = tuple(sorted([other_face[j], other_face[(j+1)%3]]))
                            if edge == other_edge:
                                edge_face_count += 1
                    
                    # Se uma aresta é compartilhada por menos de 2 faces, a face não é interior
                    if edge_face_count < 2:
                        is_interior = False
                        break
                
                if is_interior:
                    faces_to_remove.add(face_idx)
            
            # Remove faces intersectantes (faces que se cruzam)
            # Detecta interseções usando bounding boxes das faces
            for i in range(len(mesh.faces)):
                if i in faces_to_remove:
                    continue
                    
                face1 = mesh.faces[i]
                vertices1 = mesh.vertices[face1]
                
                for j in range(i + 1, len(mesh.faces)):
                    if j in faces_to_remove:
                        continue
                        
                    face2 = mesh.faces[j]
                    vertices2 = mesh.vertices[face2]
                    
                    # Verifica se as faces compartilham vértices
                    shared_vertices = set(face1) & set(face2)
                    if len(shared_vertices) >= 2:
                        # Faces compartilham arestas, verifica se há interseção
                        # Calcula bounding box das faces
                        bbox1_min = np.min(vertices1, axis=0)
                        bbox1_max = np.max(vertices1, axis=0)
                        bbox2_min = np.min(vertices2, axis=0)
                        bbox2_max = np.max(vertices2, axis=0)
                        
                        # Se os bounding boxes se intersectam, pode haver interseção
                        if (bbox1_min < bbox2_max).all() and (bbox2_min < bbox1_max).all():
                            # Verifica se há sobreposição significativa
                            overlap_volume = np.prod(np.minimum(bbox1_max, bbox2_max) - np.maximum(bbox1_min, bbox2_min))
                            bbox1_volume = np.prod(bbox1_max - bbox1_min)
                            bbox2_volume = np.prod(bbox2_max - bbox2_min)
                            
                            # Se há sobreposição significativa, remove uma das faces
                            if overlap_volume > 0.1 * min(bbox1_volume, bbox2_volume):
                                faces_to_remove.add(j)
            
            if not faces_to_remove:
                print("Nenhuma face interior ou intersectante encontrada")
                return
            
            print(f"Removendo {len(faces_to_remove)} faces interiores/intersectantes")
            
            # Remove as faces marcadas
            remaining_faces = [face for i, face in enumerate(mesh.faces) if i not in faces_to_remove]
            
            if not remaining_faces:
                QMessageBox.warning(self, 'Remoção de Faces', 'Todas as faces foram removidas. Operação cancelada.')
                return
            
            # Cria nova malha sem as faces removidas
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=remaining_faces, process=True)
            
            # Atualiza a malha reparada
            self.mesh_reparada = new_mesh
            
            # Atualiza visualização
            self.gl_reparada.clear()
            item = create_glmeshitem(new_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(new_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, new_mesh)
            
            print(f"Faces interiores/intersectantes removidas com sucesso. "
                  f"Faces restantes: {len(new_mesh.faces)}")
            
        except Exception as e:
            print(f"Erro ao remover faces interiores/intersectantes: {e}")
            import traceback
            traceback.print_exc()

    def edge_split_dialog(self):
        """Diálogo para configurar o Edge Split Modifier"""
        if self.mesh_reparada is None:
            return
        
        try:
            angle_limit, ok = QInputDialog.getDouble(
                self, 'Edge Split Modifier', 
                'Ângulo limite para divisão de arestas (graus):',
                value=30.0, min=0.0, max=180.0, decimals=1
            )
            
            if ok:
                self.edge_split_modifier(angle_limit)
                
        except Exception as e:
            print(f"Erro no diálogo Edge Split: {e}")
            import traceback
            traceback.print_exc()

    def edge_split_modifier(self, angle_limit):
        """Aplica o Edge Split Modifier na malha"""
        if self.mesh_reparada is None:
            return
        
        try:
            print(f"Aplicando Edge Split Modifier com ângulo limite: {angle_limit}°")
            
            # Converte ângulo para radianos
            angle_rad = np.radians(angle_limit)
            
            # Calcula normais das faces
            face_normals = self.mesh_reparada.face_normals
            
            # Encontra arestas que precisam ser divididas
            edges_to_split = set()
            
            # Para cada face, verifica suas arestas
            for face_idx, face in enumerate(self.mesh_reparada.faces):
                face_normal = face_normals[face_idx]
                
                # Para cada aresta da face
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    
                    # Encontra faces adjacentes a esta aresta
                    adjacent_faces = []
                    for other_face_idx, other_face in enumerate(self.mesh_reparada.faces):
                        if other_face_idx != face_idx:
                            # Verifica se a aresta está presente na outra face
                            for j in range(3):
                                other_edge = tuple(sorted([other_face[j], other_face[(j+1)%3]]))
                                if edge == other_edge:
                                    adjacent_faces.append(other_face_idx)
                                    break
                    
                    # Se há faces adjacentes, verifica o ângulo
                    if adjacent_faces:
                        for adj_face_idx in adjacent_faces:
                            adj_normal = face_normals[adj_face_idx]
                            
                            # Calcula ângulo entre as normals
                            dot_product = np.dot(face_normal, adj_normal)
                            dot_product = np.clip(dot_product, -1.0, 1.0)
                            angle = np.arccos(dot_product)
                            
                            # Se o ângulo é maior que o limite, marca para divisão
                            if angle > angle_rad:
                                edges_to_split.add(edge)
                                break
            
            if not edges_to_split:
                print("Nenhuma aresta precisa ser dividida")
                return
            
            print(f"Dividindo {len(edges_to_split)} arestas")
            
            # Cria nova malha com vértices duplicados nas arestas divididas
            new_vertices = list(self.mesh_reparada.vertices)
            new_faces = []
            
            # Mapeia vértices originais para novos vértices
            vertex_mapping = {}
            
            for face_idx, face in enumerate(self.mesh_reparada.faces):
                new_face = []
                
                for i in range(3):
                    vertex_idx = face[i]
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    
                    # Se a aresta deve ser dividida, cria novo vértice
                    if edge in edges_to_split:
                        # Cria novo vértice duplicado
                        new_vertex_idx = len(new_vertices)
                        new_vertices.append(self.mesh_reparada.vertices[vertex_idx])
                        
                        # Mapeia o vértice original para o novo
                        if vertex_idx not in vertex_mapping:
                            vertex_mapping[vertex_idx] = {}
                        vertex_mapping[vertex_idx][face_idx] = new_vertex_idx
                        
                        new_face.append(new_vertex_idx)
                    else:
                        # Usa vértice existente ou mapeado
                        if vertex_idx in vertex_mapping and face_idx in vertex_mapping[vertex_idx]:
                            new_face.append(vertex_mapping[vertex_idx][face_idx])
                        else:
                            new_face.append(vertex_idx)
                
                new_faces.append(new_face)
            
            # Cria nova malha
            new_mesh = trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces))
            
            # Atualiza a malha reparada
            self.mesh_reparada = new_mesh
            
            # Atualiza visualização
            self.gl_reparada.clear()
            item = create_glmeshitem(new_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(new_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, new_mesh)
            
            print(f"Edge Split Modifier aplicado com sucesso. "
                  f"Vértices: {len(self.mesh_reparada.vertices)}, "
                  f"Faces: {len(self.mesh_reparada.faces)}")
            
        except Exception as e:
            print(f"Erro no Edge Split Modifier: {e}")
            import traceback
            traceback.print_exc()

    def subdivision_surface_dialog(self):
        """Diálogo para configurar a Subdivision Surface"""
        if self.mesh_reparada is None:
            return
        
        try:
            iterations, ok = QInputDialog.getInt(
                self, 'Subdivision Surface', 
                'Número de iterações de Subdivision:',
                value=1, min=1, max=10, decimals=0
            )
            
            if ok:
                self.subdivision_surface(iterations)
                
        except Exception as e:
            print(f"Erro no diálogo Subdivision Surface: {e}")
            import traceback
            traceback.print_exc()

    def subdivision_surface(self, iterations):
        """Aplica a Subdivision Surface na malha"""
        if self.mesh_reparada is None:
            return
        
        try:
            print(f"Aplicando Subdivision Surface com {iterations} iterações")
            
            mesh = self.mesh_reparada.copy()
            
            # Aplica a subdivisão de superfície
            # PyMeshLab tem uma função de subdivisão de superfície, mas a implementação
            # pode variar. Aqui, vamos usar a subdivisão de Catmull-Clark, que é
            # uma das mais populares e robustas.
            # A subdivisão de Catmull-Clark adiciona vértices nas arestas e faces,
            # e recalcula as normais.
            
            # Primeiro, adiciona vértices nas arestas
            new_verts = []
            for v in mesh.vertices:
                new_verts.append(v)
            
            # Adiciona vértices nas faces
            for f in mesh.faces:
                # Catmull-Clark: adiciona um vértice no centro da face
                center = np.mean(mesh.vertices[f], axis=0)
                new_verts.append(center)
            
            # Adiciona vértices nas arestas
            for e in mesh.edges:
                # Catmull-Clark: adiciona um vértice no meio da aresta
                v0 = mesh.vertices[e[0]]
                v1 = mesh.vertices[e[1]]
                new_verts.append((v0 + v1) / 2)
            
            # Atualiza a malha com os novos vértices
            new_mesh = trimesh.Trimesh(vertices=np.array(new_verts), faces=mesh.faces, process=True)
            
            # Recalcula as normais após a subdivisão
            new_mesh.fix_normals()
            
            # Aplica a subdivisão recursivamente
            for _ in range(iterations):
                # Adiciona vértices nas arestas
                new_verts = []
                for v in new_mesh.vertices:
                    new_verts.append(v)
                
                # Adiciona vértices nas faces
                for f in new_mesh.faces:
                    center = np.mean(new_mesh.vertices[f], axis=0)
                    new_verts.append(center)
                
                # Adiciona vértices nas arestas
                for e in new_mesh.edges:
                    v0 = new_mesh.vertices[e[0]]
                    v1 = new_mesh.vertices[e[1]]
                    new_verts.append((v0 + v1) / 2)
                
                new_mesh = trimesh.Trimesh(vertices=np.array(new_verts), faces=new_mesh.faces, process=True)
                new_mesh.fix_normals()
            
            self.mesh_reparada = new_mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(new_mesh, color=(0.1, 0.8, 0.1, 1))
            self.gl_reparada.addItem(item)
            self.analisar_malha(new_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, new_mesh)
            
            print(f"Subdivision Surface aplicada com sucesso com {iterations} iterações. "
                  f"Vértices: {len(new_mesh.vertices)}, Faces: {len(new_mesh.faces)}")
            
        except Exception as e:
            print(f"Erro ao aplicar Subdivision Surface: {e}")
            import traceback
            traceback.print_exc()

    def solidify_modifier_dialog(self):
        """Diálogo para configurar o Solidify Modifier"""
        if self.mesh_reparada is None:
            return
        
        try:
            thickness, ok = QInputDialog.getDouble(
                self, 'Solidify Modifier', 
                'Espessura da casca (0.0 para remover casca):',
                value=0.0, min=0.0, max=100.0, decimals=6
            )
            
            if ok:
                self.solidify_modifier(thickness)
                
        except Exception as e:
            print(f"Erro no diálogo Solidify Modifier: {e}")
            import traceback
            traceback.print_exc()

    def solidify_modifier(self, thickness):
        """Aplica o Solidify Modifier na malha criando uma casca espessa"""
        if self.mesh_reparada is None:
            return
        
        try:
            print(f"Aplicando Solidify Modifier com espessura: {thickness}")
            
            if thickness <= 0:
                print("Espessura deve ser maior que 0 para criar casca")
                return
            
            mesh = self.mesh_reparada.copy()
            
            # Calcula as normais dos vértices para direção da casca
            mesh.vertex_normals = mesh.vertex_normals
            
            # Cria vértices externos deslocados na direção das normais
            outer_vertices = mesh.vertices + mesh.vertex_normals * thickness
            
            # Combina vértices originais e externos
            all_vertices = np.vstack([mesh.vertices, outer_vertices])
            
            # Cria faces para conectar a malha interna com a externa
            new_faces = []
            
            # Adiciona as faces originais
            new_faces.extend(mesh.faces)
            
            # Adiciona faces externas (invertidas para manter orientação correta)
            for face in mesh.faces:
                # Face externa com vértices deslocados
                outer_face = [v + len(mesh.vertices) for v in face]
                # Inverte a ordem para manter orientação
                outer_face = [outer_face[0], outer_face[2], outer_face[1]]
                new_faces.append(outer_face)
            
            # Cria faces laterais para conectar as bordas
            for edge in mesh.edges_unique:
                # Encontra faces que compartilham esta aresta
                edge_faces = []
                for face_idx, face in enumerate(mesh.faces):
                    for i in range(3):
                        v1, v2 = face[i], face[(i + 1) % 3]
                        if (v1 == edge[0] and v2 == edge[1]) or (v1 == edge[1] and v2 == edge[0]):
                            edge_faces.append(face_idx)
                
                if len(edge_faces) == 1:  # Aresta de borda
                    # Cria duas faces triangulares para conectar a aresta interna com a externa
                    v1_in, v2_in = edge[0], edge[1]
                    v1_out, v2_out = v1_in + len(mesh.vertices), v2_in + len(mesh.vertices)
                    
                    # Primeira face triangular
                    new_faces.append([v1_in, v2_in, v1_out])
                    # Segunda face triangular
                    new_faces.append([v2_in, v2_out, v1_out])
            
            # Cria nova malha com casca
            new_mesh = trimesh.Trimesh(vertices=all_vertices, faces=new_faces, process=True)
            new_mesh.fix_normals()
            
            self.mesh_reparada = new_mesh
            self.gl_reparada.clear()
            item = create_glmeshitem(new_mesh, color=(0.8, 0.1, 0.1, 1))  # Vermelho para destacar
            self.gl_reparada.addItem(item)
            self.analisar_malha(new_mesh, self.label_analise_reparada)
            self.centralizar_camera(self.gl_reparada, new_mesh)
            
            print(f"Solidify Modifier aplicado com sucesso. "
                  f"Vértices: {len(mesh.vertices)} → {len(all_vertices)}, "
                  f"Faces: {len(mesh.faces)} → {len(new_faces)}")
            
        except Exception as e:
            print(f"Erro ao aplicar Solidify Modifier: {e}")
            import traceback
            traceback.print_exc()

    def estatisticas_malha_dialog(self):
        """Diálogo para mostrar estatísticas detalhadas da malha"""
        if self.mesh_reparada is None:
            QMessageBox.information(self, 'Estatísticas', 'Carregue e repare uma malha primeiro.')
            return
        
        try:
            self.estatisticas_malha()
        except Exception as e:
            print(f"Erro ao gerar estatísticas: {e}")
            import traceback
            traceback.print_exc()

    def estatisticas_malha(self):
        """Gera estatísticas detalhadas da malha incluindo N-gons, triângulos e pólos"""
        if self.mesh_reparada is None:
            return
        
        try:
            mesh = self.mesh_reparada
            print("=== ESTATÍSTICAS DETALHADAS DA MALHA ===")
            
            # Estatísticas básicas
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            num_edges = len(mesh.edges_unique)
            
            print(f"Vértices: {num_vertices}")
            print(f"Faces: {num_faces}")
            print(f"Arestas únicas: {num_edges}")
            
            # Análise de tipos de faces (N-gons)
            face_vertex_counts = {}
            for face in mesh.faces:
                num_verts = len(face)
                face_vertex_counts[num_verts] = face_vertex_counts.get(num_verts, 0) + 1
            
            print("\n=== TIPOS DE FACES ===")
            for num_verts, count in sorted(face_vertex_counts.items()):
                if num_verts == 3:
                    print(f"Triângulos: {count}")
                elif num_verts == 4:
                    print(f"Quads: {count}")
                else:
                    print(f"{num_verts}-gons: {count}")
            
            # Análise de pólos (vértices com número anormal de arestas)
            vertex_edge_counts = {}
            for edge in mesh.edges:
                for vertex in edge:
                    vertex_edge_counts[vertex] = vertex_edge_counts.get(vertex, 0) + 1
            
            pole_vertices = []
            for vertex, edge_count in vertex_edge_counts.items():
                if edge_count != 4:  # Vértices com 4 arestas são normais em malhas quad
                    pole_vertices.append((vertex, edge_count))
            
            print(f"\n=== ANÁLISE DE PÓLOS ===")
            print(f"Vértices com 4 arestas (normais): {len([v for v, c in pole_vertices if c == 4])}")
            print(f"Vértices com 3 arestas (pólos): {len([v for v, c in pole_vertices if c == 3])}")
            print(f"Vértices com 5+ arestas (pólos): {len([v for v, c in pole_vertices if c >= 5])}")
            
            # Análise de topologia
            print(f"\n=== TOPOLOGIA ===")
            print(f"Malha fechada (watertight): {mesh.is_watertight}")
            print(f"Malha orientável: {mesh.is_winding_consistent}")
            print(f"Volume: {mesh.volume:.6f}" if hasattr(mesh, 'volume') else "Volume: N/A")
            print(f"Área da superfície: {mesh.area:.6f}")
            
            # Análise de qualidade
            print(f"\n=== QUALIDADE DA MALHA ===")
            
            # Faces degeneradas (área muito pequena)
            areas = mesh.area_faces
            degenerate_faces = np.sum(areas < 1e-12)
            print(f"Faces degeneradas (área < 1e-12): {degenerate_faces}")
            
            # Vértices duplicados
            unique_vertices = np.unique(mesh.vertices, axis=0)
            duplicate_vertices = num_vertices - len(unique_vertices)
            print(f"Vértices duplicados: {duplicate_vertices}")
            
            # Arestas abertas (buracos)
            if hasattr(mesh, 'edges_open'):
                open_edges = len(mesh.edges_open)
                print(f"Arestas abertas (buracos): {open_edges}")
            else:
                print("Arestas abertas: N/A")
            
            # Faces não-manifold
            if hasattr(mesh, 'faces_nonmanifold'):
                nonmanifold_faces = len(mesh.faces_nonmanifold)
                print(f"Faces não-manifold: {nonmanifold_faces}")
            else:
                print("Faces não-manifold: N/A")
            
            # Análise de distribuição de faces
            print(f"\n=== DISTRIBUIÇÃO DE FACES ===")
            if len(areas) > 0:
                print(f"Área média das faces: {np.mean(areas):.6f}")
                print(f"Área mínima das faces: {np.min(areas):.6f}")
                print(f"Área máxima das faces: {np.max(areas):.6f}")
                print(f"Desvio padrão das áreas: {np.std(areas):.6f}")
            
            # Análise de ângulos das faces
            print(f"\n=== ANÁLISE DE ÂNGULOS ===")
            angles = []
            for face in mesh.faces:
                if len(face) == 3:  # Apenas triângulos
                    v1, v2, v3 = mesh.vertices[face]
                    a = np.linalg.norm(v2 - v1)
                    b = np.linalg.norm(v3 - v2)
                    c = np.linalg.norm(v1 - v3)
                    
                    # Lei dos cossenos
                    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
                    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
                    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
                    
                    # Converte para graus
                    angles.extend([np.arccos(np.clip(cos_A, -1, 1)) * 180 / np.pi,
                                 np.arccos(np.clip(cos_B, -1, 1)) * 180 / np.pi,
                                 np.arccos(np.clip(cos_C, -1, 1)) * 180 / np.pi])
            
            if angles:
                angles = np.array(angles)
                print(f"Ângulo mínimo: {np.min(angles):.2f}°")
                print(f"Ângulo máximo: {np.max(angles):.2f}°")
                print(f"Ângulo médio: {np.mean(angles):.2f}°")
                
                # Conta ângulos muito agudos ou obtusos
                acute_angles = np.sum(angles < 30)
                obtuse_angles = np.sum(angles > 150)
                print(f"Ângulos muito agudos (< 30°): {acute_angles}")
                print(f"Ângulos muito obtusos (> 150°): {obtuse_angles}")
            
            # Recomendações baseadas na análise
            print(f"\n=== RECOMENDAÇÕES ===")
            if degenerate_faces > 0:
                print("⚠️  Considere remover faces degeneradas")
            if duplicate_vertices > 0:
                print("⚠️  Considere soldar vértices duplicados")
            if 'open_edges' in locals() and open_edges > 0:
                print("⚠️  Considere preencher buracos")
            if 'nonmanifold_faces' in locals() and nonmanifold_faces > 0:
                print("⚠️  Considere remover faces não-manifold")
            if 'acute_angles' in locals() and acute_angles > 0:
                print("⚠️  Considere suavizar ângulos muito agudos")
            
            print("=== FIM DAS ESTATÍSTICAS ===\n")
            
            # Mostra diálogo com resumo
            QMessageBox.information(self, 'Estatísticas da Malha', 
                f"Estatísticas geradas com sucesso!\n\n"
                f"Vértices: {num_vertices}\n"
                f"Faces: {num_faces}\n"
                f"Triângulos: {face_vertex_counts.get(3, 0)}\n"
                f"Quads: {face_vertex_counts.get(4, 0)}\n"
                f"N-gons: {sum(count for verts, count in face_vertex_counts.items() if verts > 4)}\n"
                f"Pólos: {len([v for v, c in pole_vertices if c != 4])}\n\n"
                f"Verifique o console para detalhes completos.")
            
        except Exception as e:
            print(f"Erro ao gerar estatísticas: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, 'Erro', f'Erro ao gerar estatísticas:\n{e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MeshRepairApp()
    window.show()
    sys.exit(app.exec_()) 