import sys
import trimesh
import pymeshfix
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSizePolicy, QDoubleSpinBox, QMainWindow, QAction, QMenuBar, QInputDialog
)
from pyqtgraph.opengl import GLViewWidget, MeshData, GLMeshItem, GLLinePlotItem
from PyQt5.QtCore import Qt
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
        self.setWindowTitle('Reparo de Malha 3D - STL/OBJ')
        self.resize(1200, 600)
        self.mesh_original = None
        self.mesh_reparada = None
        # Widgets centrais
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.gl_original = GLViewWidget()
        self.gl_reparada = GLViewWidget()
        self.gl_original.setCameraPosition(distance=200)
        self.gl_reparada.setCameraPosition(distance=200)
        self.gl_original.setBackgroundColor('w')
        self.gl_reparada.setBackgroundColor('w')
        self.gl_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl_reparada.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_original = QLabel('Original')
        self.label_original.setAlignment(Qt.AlignHCenter)
        self.label_reparada = QLabel('Reparada')
        self.label_reparada.setAlignment(Qt.AlignHCenter)
        self.label_analise = QLabel('')
        self.label_analise.setAlignment(Qt.AlignHCenter)
        self.label_analise_reparada = QLabel('')
        self.label_analise_reparada.setAlignment(Qt.AlignHCenter)
        # Botões principais
        self.btn_abrir = QPushButton('Abrir STL/OBJ')
        self.btn_reparar = QPushButton('Reparar Malha')
        self.btn_salvar = QPushButton('Salvar Malha Reparada')
        self.btn_abrir.clicked.connect(self.abrir_arquivo)
        self.btn_reparar.clicked.connect(self.reparar_malha)
        self.btn_salvar.clicked.connect(self.salvar_malha)
        self.btn_abrir.setEnabled(True)
        self.btn_reparar.setEnabled(False)
        self.btn_salvar.setEnabled(False)
        # Layout principal
        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_abrir)
        btn_layout.addWidget(self.btn_reparar)
        btn_layout.addWidget(self.btn_salvar)
        layout.addLayout(btn_layout)
        vis_layout = QHBoxLayout()
        vis_left = QVBoxLayout()
        vis_left.addWidget(self.gl_original)
        vis_left.addWidget(self.label_original)
        vis_left.addWidget(self.label_analise)
        vis_right = QVBoxLayout()
        vis_right.addWidget(self.gl_reparada)
        vis_right.addWidget(self.label_reparada)
        vis_right.addWidget(self.label_analise_reparada)
        vis_layout.addLayout(vis_left)
        vis_layout.addLayout(vis_right)
        layout.addLayout(vis_layout)
        central_widget.setLayout(layout)
        # Barra de menu
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        # Menu Topologia/Geometria
        self.menu_topologia = self.menu_bar.addMenu('Topologia/Geometria')
        self.action_remove_doubles = QAction('Remover Duplicados', self)
        self.action_remove_doubles.triggered.connect(self.remover_duplicados_dialog)
        self.menu_topologia.addAction(self.action_remove_doubles)
        self.action_suavizar = QAction('Suavizar Malha Reparada', self)
        self.action_suavizar.triggered.connect(self.suavizar_malha)
        self.menu_topologia.addAction(self.action_suavizar)
        self.action_normals_out = QAction('Recalcular Normais para Fora', self)
        self.action_normals_out.triggered.connect(lambda: self.recalcular_normais('out'))
        self.menu_topologia.addAction(self.action_normals_out)
        self.action_normals_in = QAction('Recalcular Normais para Dentro', self)
        self.action_normals_in.triggered.connect(lambda: self.recalcular_normais('in'))
        self.menu_topologia.addAction(self.action_normals_in)
        self.action_fill_holes = QAction('Preencher Buracos / Tornar Manifold', self)
        self.action_fill_holes.triggered.connect(self.preencher_buracos)
        self.menu_topologia.addAction(self.action_fill_holes)
        self.action_fill_holes.setEnabled(False)
        self.action_remove_nonmanifold = QAction('Remover Geometria Não-Manifold', self)
        self.action_remove_nonmanifold.triggered.connect(self.remover_nao_manifold)
        self.menu_topologia.addAction(self.action_remove_nonmanifold)
        self.action_remove_nonmanifold.setEnabled(False)
        self.action_decimate = QAction('Simplificar Malha (Decimate)', self)
        self.action_decimate.triggered.connect(self.simplificar_malha_dialog)
        self.menu_topologia.addAction(self.action_decimate)
        self.action_decimate.setEnabled(False)
        self.action_triangulate = QAction('Triangular Faces (Triangulate)', self)
        self.action_triangulate.triggered.connect(self.triangulate_faces)
        self.menu_topologia.addAction(self.action_triangulate)
        self.action_triangulate.setEnabled(False)
        self.action_quadrangulate = QAction('Quadrangular Faces (Quadrangulate)', self)
        self.action_quadrangulate.triggered.connect(self.quadrangulate_faces)
        self.menu_topologia.addAction(self.action_quadrangulate)
        self.action_quadrangulate.setEnabled(False)
        self.action_remove_degenerate = QAction('Remover Faces Degeneradas/Área Zero', self)
        self.action_remove_degenerate.triggered.connect(self.remover_faces_degeneradas)
        self.menu_topologia.addAction(self.action_remove_degenerate)
        self.action_remove_degenerate.setEnabled(False)
        self.action_remesh = QAction('Remesh (Voxel)', self)
        self.action_remesh.triggered.connect(self.remesh_voxel_dialog)
        self.menu_topologia.addAction(self.action_remesh)
        self.action_remesh.setEnabled(False)
        # Remover Quadriflow do menu e do código
        # self.action_remesh_quadriflow = QAction('Remesh (Quadriflow)', self)
        # self.action_remesh_quadriflow.triggered.connect(self.remesh_quadriflow_dialog)
        # self.menu_topologia.addAction(self.action_remesh_quadriflow)
        # self.action_remesh_quadriflow.setEnabled(False)
        self.action_remesh_surface = QAction('Remesh (Surface)', self)
        self.action_remesh_surface.triggered.connect(self.remesh_surface_dialog)
        self.menu_topologia.addAction(self.action_remesh_surface)
        self.action_remesh_surface.setEnabled(False)
        self.action_listar_metodos = QAction('Listar Métodos PyMeshLab', self)
        self.action_listar_metodos.triggered.connect(self.listar_metodos_pymeshlab)
        self.menu_topologia.addAction(self.action_listar_metodos)
        self.action_exportar_metodos = QAction('Exportar Métodos PyMeshLab', self)
        self.action_exportar_metodos.triggered.connect(self.exportar_metodos_pymeshlab)
        self.menu_topologia.addAction(self.action_exportar_metodos)
        self.action_exportar_atributos = QAction('Exportar Atributos PyMeshLab', self)
        self.action_exportar_atributos.triggered.connect(self.exportar_atributos_pymeshlab)
        self.menu_topologia.addAction(self.action_exportar_atributos)
        self.action_auto_retopo = QAction('Auto Retopology', self)
        self.action_auto_retopo.triggered.connect(self.auto_retopology_dialog)
        self.menu_topologia.addAction(self.action_auto_retopo)
        self.action_auto_retopo.setEnabled(False)
        # Novo menu Sombreamento
        self.menu_sombreamento = self.menu_bar.addMenu('Sombreamento')
        self.action_shade_smooth = QAction('Shade Smooth', self)
        self.action_shade_smooth.triggered.connect(self.shade_smooth)
        self.menu_sombreamento.addAction(self.action_shade_smooth)
        self.action_shade_smooth.setEnabled(False)
        self.action_shade_flat = QAction('Shade Flat', self)
        self.action_shade_flat.triggered.connect(self.shade_flat)
        self.menu_sombreamento.addAction(self.action_shade_flat)
        self.action_shade_flat.setEnabled(False)
        self.action_auto_smooth = QAction('Auto Smooth', self)
        self.action_auto_smooth.triggered.connect(self.auto_smooth_dialog)
        self.menu_sombreamento.addAction(self.action_auto_smooth)
        self.action_auto_smooth.setEnabled(False)
        self.action_transfer_normals = QAction('Transferir Normais da Original', self)
        self.action_transfer_normals.triggered.connect(self.transferir_normais)
        self.menu_sombreamento.addAction(self.action_transfer_normals)
        self.action_transfer_normals.setEnabled(False)
        # Menu Visualização
        self.menu_visualizacao = self.menu_bar.addMenu('Visualização')
        self.action_reset_original = QAction('Resetar Visualização Original', self)
        self.action_reset_original.triggered.connect(lambda: self.centralizar_camera(self.gl_original, self.mesh_original))
        self.menu_visualizacao.addAction(self.action_reset_original)
        self.action_reset_reparada = QAction('Resetar Visualização Reparada', self)
        self.action_reset_reparada.triggered.connect(lambda: self.centralizar_camera(self.gl_reparada, self.mesh_reparada))
        self.menu_visualizacao.addAction(self.action_reset_reparada)
        # Habilitar/desabilitar ações conforme contexto
        self.action_remove_doubles.setEnabled(False)
        self.action_suavizar.setEnabled(False)
        self.action_reset_original.setEnabled(True)
        self.action_reset_reparada.setEnabled(False)
        self.action_normals_out.setEnabled(False)
        self.action_normals_in.setEnabled(False)
        self.action_fill_holes.setEnabled(False)
        self.action_remove_nonmanifold.setEnabled(False)
        self.action_decimate.setEnabled(False)
        self.action_triangulate.setEnabled(False)
        self.action_quadrangulate.setEnabled(False)
        self.action_remove_degenerate.setEnabled(False)
        self.action_remesh.setEnabled(False)
        # self.action_remesh_quadriflow.setEnabled(False) # Removido
        self.action_remesh_surface.setEnabled(False)
        self.action_auto_retopo.setEnabled(False)
        self.action_shade_smooth.setEnabled(False)
        self.action_shade_flat.setEnabled(False)
        self.action_auto_smooth.setEnabled(False)
        self.action_transfer_normals.setEnabled(False)

    def abrir_arquivo(self):
        from PyQt5.QtWidgets import QMessageBox
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir arquivo STL/OBJ', '', 'Malhas 3D (*.stl *.obj)')
        if fname:
            mesh = trimesh.load(fname, process=False)
            # Tentar processar e corrigir problemas leves
            try:
                mesh.process(validate=True)
            except Exception as e:
                QMessageBox.warning(self, 'Aviso', f'Problemas ao processar a malha original.\n{e}')
            mesh = centralizar_na_origem(mesh)
            self.mesh_original = mesh
            self.gl_original.clear()
            self.gl_reparada.clear()
            try:
                item = create_glmeshitem(self.mesh_original, color=(0.1, 0.3, 1, 1))  # azul mais forte
                self.gl_original.addItem(item)
            except Exception as e:
                QMessageBox.warning(self, 'Erro ao Renderizar', f'Não foi possível renderizar a malha original.\nA malha pode estar corrompida ou precisar de reparo.\n{e}')
            self.highlight_holes(self.mesh_original)
            self.highlight_nonmanifold_faces(self.mesh_original)
            self.analisar_malha(self.mesh_original, self.label_analise)
            self.label_analise_reparada.setText('')
            self.mesh_reparada = None
            self.btn_reparar.setEnabled(True)
            self.btn_salvar.setEnabled(False)
            self.action_remove_doubles.setEnabled(False)
            self.action_suavizar.setEnabled(False)
            self.action_reset_original.setEnabled(True)
            self.action_reset_reparada.setEnabled(False)
            self.action_normals_out.setEnabled(False)
            self.action_normals_in.setEnabled(False)
            self.action_fill_holes.setEnabled(False)
            self.action_remove_nonmanifold.setEnabled(False)
            self.action_decimate.setEnabled(False)
            self.action_triangulate.setEnabled(False)
            self.action_quadrangulate.setEnabled(False)
            self.action_remove_degenerate.setEnabled(False)
            self.action_remesh.setEnabled(False)
            self.action_remesh_surface.setEnabled(False)
            self.action_auto_retopo.setEnabled(False)
            self.action_shade_smooth.setEnabled(False)
            self.action_shade_flat.setEnabled(False)
            self.action_auto_smooth.setEnabled(False)
            self.action_transfer_normals.setEnabled(False)
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
        mesh = self.mesh_original
        if mesh.is_watertight:
            mesh_reparada = mesh.copy()
        else:
            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(verbose=True)
            mesh_reparada = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
        mesh_reparada = centralizar_na_origem(mesh_reparada)
        self.mesh_reparada = mesh_reparada
        self.gl_reparada.clear()
        item = create_glmeshitem(mesh_reparada, color=(0.1, 0.8, 0.1, 1))  # verde mais forte
        self.gl_reparada.addItem(item)
        self.analisar_malha(mesh_reparada, self.label_analise_reparada)
        self.btn_salvar.setEnabled(True)
        self.action_remove_doubles.setEnabled(True)
        self.action_suavizar.setEnabled(True)
        self.action_reset_reparada.setEnabled(True)
        self.action_normals_out.setEnabled(True)
        self.action_normals_in.setEnabled(True)
        self.action_fill_holes.setEnabled(True)
        self.action_remove_nonmanifold.setEnabled(True)
        self.action_decimate.setEnabled(True)
        self.action_triangulate.setEnabled(True)
        self.action_quadrangulate.setEnabled(True)
        self.action_remove_degenerate.setEnabled(True)
        self.action_remesh.setEnabled(True)
        self.action_remesh_surface.setEnabled(True)
        self.action_auto_retopo.setEnabled(True)
        self.action_shade_smooth.setEnabled(True)
        self.action_shade_flat.setEnabled(True)
        self.action_auto_smooth.setEnabled(True)
        self.action_transfer_normals.setEnabled(True)
        self.centralizar_camera(self.gl_reparada, mesh_reparada)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MeshRepairApp()
    window.show()
    sys.exit(app.exec_()) 