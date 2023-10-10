from collections import defaultdict
from pathlib import Path
from PIL import Image
import os


class Preprocess():
    def start(self, input_folder, output_folder):
        print ("[+] Start Data Pre-processing")

        if os.path.exists(output_folder):
            print (f"[-] {output_folder} already exists. Choose an empty one.")
            return
        else:
            os.mkdir(output_folder)

        shapes = defaultdict(list)
        folder_names = []
        
        for file in Path(input_folder).iterdir():
            if not file.is_file():
                continue
            img = Image.open(file)
            shapes[img.size].append(file)
        
        len_shapes = len(shapes.keys())
        print (f"[+] In this collection of bitmap caches, there are {len_shapes} sizes:")
        
        for k, v in shapes.items():
            print (f"     * {len(v)} images are {k}.")
            width, height = k
            
            folder = Path(f"{output_folder}/{height}x{width}")
            folder_names.append(folder.name)
            if not folder.exists():
                folder.mkdir()

            for src in v:
                dst = folder / f"{src.name.replace('{cached_file}_','')}"
                os.symlink(src.resolve(), dst.resolve())
        
        print (f"[+] Therefore, they have been split into {len_shapes} pool(s) with the following folder names: {folder_names}.")

