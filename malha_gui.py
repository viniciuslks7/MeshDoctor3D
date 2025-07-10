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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MeshRepairApp()
    window.show()
    sys.exit(app.exec_()) 