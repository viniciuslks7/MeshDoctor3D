import sys
import os
import trimesh
import pymeshfix

# Função para reparar a malha
def reparar_malha(input_path, output_path=None):
    # Carrega a malha
    mesh = trimesh.load(input_path)
    if not mesh.is_watertight:
        print("Malha não é watertight. Tentando reparar...")
        # Repara a malha
        meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        meshfix.repair(verbose=True)
        # Cria uma nova malha trimesh com os dados reparados
        mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
    else:
        print("Malha já é watertight!")
    # Salva a malha reparada
    if not output_path:
        nome, ext = os.path.splitext(input_path)
        output_path = f"{nome}_reparado{ext}"
    mesh.export(output_path)
    print(f"Malha reparada salva em: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python reparar_malha.py arquivo.stl [saida.stl]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    reparar_malha(input_file, output_file) 