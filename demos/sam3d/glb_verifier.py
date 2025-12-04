#!/usr/bin/env python3
import sys
import struct
import json
from collections import Counter
from pathlib import Path


GLTF_MAGIC = b'glTF'
CHUNK_TYPE_JSON = 0x4E4F534A  # 'JSON'
CHUNK_TYPE_BIN = 0x004E4942   # 'BIN\0'


class GlbError(RuntimeError):
    pass


def read_glb_header(f):
    header = f.read(12)
    if len(header) != 12:
        raise GlbError("File too short for GLB header (need 12 bytes).")

    magic, version, length = struct.unpack("<4sII", header)

    if magic != GLTF_MAGIC:
        raise GlbError(f"Bad magic: {magic!r}, expected b'glTF'.")

    return version, length


def read_glb_chunks(f):
    """
    Returns (json_bytes, bin_bytes or None, raw_chunks_info)
    """
    json_chunk = None
    bin_chunk = None
    chunks_info = []

    while True:
        header = f.read(8)
        if not header:
            break
        if len(header) != 8:
            raise GlbError("Truncated chunk header (need 8 bytes).")

        chunk_len, chunk_type = struct.unpack("<II", header)
        data = f.read(chunk_len)
        if len(data) != chunk_len:
            raise GlbError("Truncated chunk data.")

        chunks_info.append((chunk_len, chunk_type))

        if chunk_type == CHUNK_TYPE_JSON:
            json_chunk = data
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = data
        else:
            # Unknown chunk type; keep moving but record it
            pass

    if json_chunk is None:
        raise GlbError("No JSON chunk found in GLB.")

    return json_chunk, bin_chunk, chunks_info


def load_gltf_json(json_bytes):
    try:
        text = json_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise GlbError(f"JSON chunk is not valid UTF-8: {e}")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise GlbError(f"JSON parse error: {e}")


def summarize_extensions(gltf):
    used = gltf.get("extensionsUsed", [])
    required = gltf.get("extensionsRequired", [])
    return used, required


def count_primitives(gltf):
    meshes = gltf.get("meshes", [])
    n_meshes = len(meshes)
    n_prims = 0

    prim_ext_counter = Counter()

    for m in meshes:
        for prim in m.get("primitives", []):
            n_prims += 1
            ext = prim.get("extensions", {})
            for k in ext.keys():
                prim_ext_counter[k] += 1

    return n_meshes, n_prims, prim_ext_counter


def count_nodes_scenes(gltf):
    return len(gltf.get("nodes", [])), len(gltf.get("scenes", [])), gltf.get("scene")


def count_skins_anims(gltf):
    return len(gltf.get("skins", [])), len(gltf.get("animations", []))


def count_materials_textures(gltf):
    return len(gltf.get("materials", [])), len(gltf.get("textures", [])), len(gltf.get("images", []))


def print_separator():
    print("-" * 60)


def analyze_glb(path: Path):
    with path.open("rb") as f:
        print_separator()
        print(f"Verifying GLB: {path}")
        print_separator()

        version, length = read_glb_header(f)
        print(f"Header:")
        print(f"  version: {version}")
        print(f"  declared length: {length} bytes")

        json_bytes, bin_bytes, chunks_info = read_glb_chunks(f)

        print("\nChunks:")
        for i, (clen, ctype) in enumerate(chunks_info):
            type_str = {
                CHUNK_TYPE_JSON: "JSON",
                CHUNK_TYPE_BIN: "BIN "
            }.get(ctype, f"UNKNOWN (0x{ctype:08X})")
            print(f"  #{i}: length={clen}, type={type_str}")

        if bin_bytes is None:
            print("WARNING: No BIN chunk found (unusual for GLB with geometry).")

        gltf = load_gltf_json(json_bytes)

        print_separator()
        print("Core glTF stats:")
        n_meshes, n_prims, prim_ext_counter = count_primitives(gltf)
        n_nodes, n_scenes, default_scene = count_nodes_scenes(gltf)
        n_skins, n_anims = count_skins_anims(gltf)
        n_mats, n_tex, n_img = count_materials_textures(gltf)

        print(f"  meshes:          {n_meshes}")
        print(f"  primitives:      {n_prims}")
        print(f"  nodes:           {n_nodes}")
        print(f"  scenes:          {n_scenes} (default scene index: {default_scene})")
        print(f"  skins:           {n_skins}")
        print(f"  animations:      {n_anims}")
        print(f"  materials:       {n_mats}")
        print(f"  textures:        {n_tex}")
        print(f"  images:          {n_img}")

        used, required = summarize_extensions(gltf)

        print_separator()
        print("Extensions:")
        print(f"  extensionsUsed:     {used if used else '[]'}")
        print(f"  extensionsRequired: {required if required else '[]'}")

        if prim_ext_counter:
            print("  Primitive-level extensions:")
            for k, v in prim_ext_counter.items():
                print(f"    {k}: {v} primitive(s)")
        else:
            print("  Primitive-level extensions: none")

        # Heuristic warnings for OVRGLTFLoader / limited loaders:
        print_separator()
        print("Compatibility warnings (heuristic):")
        any_warning = False

        # 1. Draco
        if "KHR_draco_mesh_compression" in used or "KHR_draco_mesh_compression" in prim_ext_counter:
            any_warning = True
            print("  [LIKELY PROBLEM] Uses KHR_draco_mesh_compression (Draco).")
            print("    Many lightweight loaders, including Meta/OVR ones, do NOT implement Draco.")
            print("    Expect missing or partial geometry there.")

        # 2. Other non-trivial extensions
        problem_like = [
            "KHR_materials_unlit",
            "KHR_materials_pbrSpecularGlossiness",
            "KHR_materials_clearcoat",
            "KHR_materials_transmission",
            "KHR_materials_volume",
            "KHR_materials_ior",
            "KHR_mesh_quantization",
            "KHR_lights_punctual",
            "KHR_texture_transform",
        ]
        for ext in used:
            if ext != "KHR_draco_mesh_compression" and ext in problem_like:
                any_warning = True
                print(f"  [POTENTIAL ISSUE] Uses extension {ext}.")
                print("    Check if your target loader supports this; some ignore it silently.")

        # 3. Multiple scenes
        if n_scenes > 1:
            any_warning = True
            print(f"  [POTENTIAL ISSUE] GLB contains {n_scenes} scenes.")
            print("    Some loaders only import the default scene; other scenes' meshes will appear 'missing'.")

        # 4. Heavy skinned / animated content
        if n_skins > 0 or n_anims > 0:
            any_warning = True
            print(f"  [POTENTIAL ISSUE] Has {n_skins} skins and {n_anims} animations.")
            print("    If your loader is basic, parts bound to skins/animations may be ignored or broken.")

        # 5. Mesh/primitive count sanity
        if n_prims == 0 or n_meshes == 0:
            any_warning = True
            print("  [LIKELY PROBLEM] No meshes/primitives reported in JSON.")
            print("    Either the file is corrupt, or all actual geometry is in an extension the loader can't see.")

        if not any_warning:
            print("  No obvious spec-level red flags detected for a basic GLB loader.")
            print("  If OVRGLTFLoader still shows a partial mesh, it's likely:")
            print("    - a loader bug or limitation not exposed via extensions, or")
            print("    - a transform/scale/scene-selection issue on the Unity side.")

        print_separator()

        analyze_primitive_accessors(gltf)

def analyze_primitive_accessors(gltf):
    accessors = gltf.get("accessors", [])
    meshes = gltf.get("meshes", [])

    if not meshes:
        print("No meshes present.")
        return

    print_separator()
    print("Primitive accessor details:")

    for mi, mesh in enumerate(meshes):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            print(f"  Mesh {mi}, Primitive {pi}:")

            # Indices
            idx_idx = prim.get("indices")
            if idx_idx is None:
                print("    indices: NONE")
            else:
                idx_acc = accessors[idx_idx]
                print(f"    indices accessor index: {idx_idx}")
                print(f"      count:          {idx_acc.get('count')}")
                print(f"      componentType:  {idx_acc.get('componentType')}  "
                      "(5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT)")
                print(f"      type:           {idx_acc.get('type')}")

                count = idx_acc.get("count", 0)
                ctype = idx_acc.get("componentType")
                if ctype == 5125:
                    print("      WARNING: uses UNSIGNED_INT indices (32-bit).")
                    print("        Many light-weight loaders only support 16-bit indices.")
                if count is not None and count > 65535:
                    print("      WARNING: index count > 65535;")
                    print("        impossible to address with 16-bit indices without splitting.")

            # Attributes (POSITION is the key one)
            attrs = prim.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            if pos_idx is not None:
                pos_acc = accessors[pos_idx]
                print(f"    POSITION accessor index: {pos_idx}")
                print(f"      count:         {pos_acc.get('count')}")
                print(f"      componentType: {pos_acc.get('componentType')}")
                print(f"      type:          {pos_acc.get('type')}")
            else:
                print("    POSITION: MISSING (invalid glTF if truly absent)")



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: verify_glb.py path/to/model.glb")
        sys.exit(1)

    path = Path(argv[0])
    if not path.is_file():
        print(f"Error: {path} is not a file.")
        sys.exit(1)

    try:
        analyze_glb(path)
    except GlbError as e:
        print_separator()
        print(f"GLB ERROR: {e}")
        print_separator()
        sys.exit(2)


if __name__ == "__main__":
    main()
