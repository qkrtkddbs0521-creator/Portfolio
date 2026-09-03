import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import os
import math
import numpy as np
import warnings
import random

# 대형 이미지 처리 제한 해제
from PIL import Image as PILImage
PILImage.MAX_IMAGE_PIXELS = None 
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class ExpertsMockupGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("전문가용 3D 목업 V26 (Perfect Shadow Flow)")
        self.root.geometry("1200x950")

        self.corner_radius = tk.IntVar(value=40)
        self.bg_color = tk.StringVar(value="#E5E5E5")
        self.shadow_offset = tk.IntVar(value=40)
        self.shadow_blur = tk.IntVar(value=60)
        self.shadow_opacity = tk.IntVar(value=90)
        self.item_gap_px = tk.IntVar(value=20)
        self.user_columns = tk.IntVar(value=5)
        self.user_scale = tk.DoubleVar(value=0.7) 
        
        self.rot_x = tk.DoubleVar(value=25.0)
        self.rot_y = tk.DoubleVar(value=-30.0)
        self.rot_z = tk.DoubleVar(value=0.0)
        
        self.images_folder = tk.StringVar(value="")
        self.preview_image = None
        self.final_mockup = None

        self.create_widgets()
        self.draw_cube()

    def create_widgets(self):
        self.frame_preview = ttk.LabelFrame(self.root, text="Mockup Preview")
        self.frame_preview.pack(pady=10, padx=20, fill="both", expand=True)
        self.lbl_preview = ttk.Label(self.frame_preview, text="🚀 목업 생성 클릭")
        self.lbl_preview.pack(expand=True)

        ctrl_main = ttk.Frame(self.root)
        ctrl_main.pack(fill="x", padx=20, pady=10)

        self.cube_canvas = tk.Canvas(ctrl_main, width=150, height=150, bg="white")
        self.cube_canvas.pack(side="left", padx=10)

        tabs = ttk.Notebook(ctrl_main)
        tabs.pack(side="left", fill="both", expand=True, padx=10)

        tab1 = ttk.Frame(tabs); tabs.add(tab1, text="Layout")
        self.add_input(tab1, "Columns", self.user_columns, 1, 50)
        self.add_input(tab1, "Gap (px)", self.item_gap_px, 0, 150)
        self.add_input(tab1, "Zoom", self.user_scale, 0.1, 2.0)
        
        tab2 = ttk.Frame(tabs); tabs.add(tab2, text="3D Rotation")
        self.add_input(tab2, "X Rot", self.rot_x, -90, 90)
        self.add_input(tab2, "Y Rot", self.rot_y, -90, 90)
        self.add_input(tab2, "Z Rot", self.rot_z, -180, 180)

        tab3 = ttk.Frame(tabs); tabs.add(tab3, text="Style")
        self.add_input(tab3, "Shadow Blur", self.shadow_blur, 0, 200)
        self.add_input(tab3, "Shadow Off", self.shadow_offset, 0, 200)
        self.add_input(tab3, "Opacity", self.shadow_opacity, 0, 255)
        self.add_input(tab3, "Rounding", self.corner_radius, 0, 200)

        btn_p = ttk.Frame(ctrl_main)
        btn_p.pack(side="right", padx=10)
        ttk.Button(btn_p, text="📁 Folder", command=self.select_folder).pack(fill="x", pady=2)
        tk.Button(btn_p, text="🎨 BG Color", command=self.choose_color).pack(fill="x", pady=2)
        ttk.Button(btn_p, text="🚀 Generate", command=self.generate_and_preview).pack(fill="x", pady=2)
        ttk.Button(btn_p, text="💾 Save", command=self.save_final_image).pack(fill="x", pady=2)

    def add_input(self, parent, label, var, f, t):
        fr = ttk.Frame(parent); fr.pack(fill="x", pady=1, padx=5)
        ttk.Label(fr, text=label, width=12).pack(side="left")
        ttk.Scale(fr, from_=f, to=t, variable=var, orient="horizontal", command=lambda e: self.draw_cube()).pack(side="left", fill="x", expand=True, padx=5)
        tk.Entry(fr, textvariable=var, width=6).pack(side="right")

    def draw_cube(self):
        self.cube_canvas.delete("all")
        c, s = 75, 50
        ax, ay, az = [math.radians(v.get()) for v in [self.rot_x, self.rot_y, self.rot_z]]
        pts = np.array([[-1,-1,-0.1],[1,-1,-0.1],[1,1,-0.1],[-1,1,-0.1],[-1,-1,0.1],[1,-1,0.1],[1,1,0.1],[-1,1,0.1]])
        rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]])
        ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
        rz = np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
        transformed = pts @ (rx @ ry @ rz).T
        proj = [(c + p[0]*s, c - p[1]*s) for p in transformed]
        faces = [([4,5,6,7], "#4180fe"), ([0,3,7,4], "#888888"), ([1,2,6,5], "#aaaaaa"), ([2,3,7,6], "#cccccc")]
        faces.sort(key=lambda x: np.mean([transformed[p][2] for p in x[0]]))
        for f_pts, color in faces:
            self.cube_canvas.create_polygon([proj[p] for p in f_pts], fill=color, outline="white")

    def select_folder(self):
        path = filedialog.askdirectory()
        if path: self.images_folder.set(path)

    def choose_color(self):
        color = colorchooser.askcolor()
        if color[1]: self.bg_color.set(color[1])

    def get_perspective_coeffs(self, src_pts, dst_pts):
        matrix = []
        for s, d in zip(src_pts, dst_pts):
            matrix.append([s[0], s[1], 1, 0, 0, 0, -d[0]*s[0], -d[0]*s[1]])
            matrix.append([0, 0, 0, s[0], s[1], 1, -d[1]*s[0], -d[1]*s[1]])
        A = np.array(matrix)
        B = np.array(dst_pts).reshape(8)
        res = np.linalg.solve(A, B)
        return res

    def generate_and_preview(self):
        src_dir = self.images_folder.get()
        if not src_dir: return
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:300]
        if not files: return

        cols = int(self.user_columns.get())
        rows = math.ceil(len(files) / cols)
        gap = self.item_gap_px.get()

        with Image.open(os.path.join(src_dir, files[0])) as sample:
            u_w = 400
            u_h = int(u_w * (sample.height / sample.width))
        
        inner_w = (cols * u_w) + ((cols - 1) * gap)
        inner_h = (rows * u_h) + ((rows - 1) * gap)
        grid_img = Image.new("RGBA", (inner_w, inner_h), (0,0,0,0))
        for i, name in enumerate(files):
            try:
                with Image.open(os.path.join(src_dir, name)) as raw:
                    img = raw.convert("RGBA").resize((u_w, u_h), Image.LANCZOS)
                    mask = Image.new("L", (u_w, u_h), 0)
                    ImageDraw.Draw(mask).rounded_rectangle((0,0,u_w,u_h), radius=self.corner_radius.get(), fill=255)
                    img.putalpha(mask)
                    grid_img.paste(img, ((i % cols) * (u_w + gap), (i // cols) * (u_h + gap)), img)
            except: continue

        # 3D Transform
        ax, ay, az = [math.radians(v.get()) for v in [self.rot_x, self.rot_y, self.rot_z]]
        rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]])
        ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
        rz = np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
        rot_mat = rx @ ry @ rz
        hw, hh = inner_w / 2, inner_h / 2
        src_corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]])
        dist = max(inner_w, inner_h) * 2.5
        projected = []
        for p_vec in src_corners:
            p = rot_mat @ p_vec
            z_f = dist / (dist - p[2])
            projected.append([p[0] * z_f, p[1] * z_f])

        projected = np.array(projected)
        min_p, max_p = np.min(projected, axis=0), np.max(projected, axis=0)
        new_w, new_h = int(max_p[0] - min_p[0]) + 40, int(max_p[1] - min_p[1]) + 40
        dst_corners = projected - min_p + [20, 20]
        coeffs = self.get_perspective_coeffs(dst_corners, [(0, 0), (inner_w, 0), (inner_w, inner_h), (0, inner_h)])
        tilted = grid_img.transform((new_w, new_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        # Scale
        scale = self.user_scale.get()
        cw, ch = int(new_w * scale), int(new_h * scale)
        content_res = tilted.resize((cw, ch), Image.LANCZOS)

        # [그림자 핵심 수정] 
        s_blur = max(1, int(self.shadow_blur.get()))
        s_off = int(self.shadow_offset.get())
        
        # 1. 그림자 마스크 생성 후, 블러가 퍼질 수 있게 레이어 자체를 미리 확장
        shadow_pad = s_blur * 4
        shadow_canvas_w = cw + shadow_pad * 2
        shadow_canvas_h = ch + shadow_pad * 2
        
        # 투명한 큰 레이어에 그림자 형태만 그림
        shadow_layer = Image.new("RGBA", (shadow_canvas_w, shadow_canvas_h), (0,0,0,0))
        s_mask = content_res.getchannel('A')
        shadow_solid = Image.new("RGBA", (cw, ch), (0,0,0,int(self.shadow_opacity.get())))
        shadow_layer.paste(shadow_solid, (shadow_pad, shadow_pad), mask=s_mask)
        
        # 2. 확장된 레이어에서 블러 적용 (이제 경계가 멀어서 안 잘림)
        shadow_final = shadow_layer.filter(ImageFilter.GaussianBlur(s_blur))
        
        # 3. 최종 도화지 계산 (그림자 오프셋 고려)
        canvas_w = shadow_canvas_w + abs(s_off)
        canvas_h = shadow_canvas_h + abs(s_off)
        self.final_mockup = Image.new("RGBA", (canvas_w, canvas_h), self.bg_color.get())
        
        # 4. 합성 (그림자를 먼저 오프셋만큼 밀어서 배치)
        off_x = s_off if s_off > 0 else 0
        off_y = s_off if s_off > 0 else 0
        img_x = shadow_pad if s_off >= 0 else shadow_pad + abs(s_off)
        img_y = shadow_pad if s_off >= 0 else shadow_pad + abs(s_off)
        
        self.final_mockup.paste(shadow_final, (off_x if s_off >=0 else 0, off_y if s_off >=0 else 0), shadow_final)
        self.final_mockup.paste(content_res, (img_x, img_y), content_res)

        self.update_preview()

    def update_preview(self):
        if self.final_mockup:
            # 프리뷰 창 크기에 맞게 썸네일 생성
            self.root.update_idletasks()
            pw, ph = self.frame_preview.winfo_width()-40, self.frame_preview.winfo_height()-60
            if pw < 100: pw, ph = 1000, 600
            
            thumb = self.final_mockup.copy()
            thumb.thumbnail((pw, ph), Image.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(thumb)
            self.lbl_preview.config(image=self.preview_image, text="")

    def save_final_image(self):
        if not self.final_mockup: return
        
        # 저장 경로 대화상자 (기본값 png)
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG (Transparency)", "*.png"), ("JPEG (No Transparency)", "*.jpg"), ("All files", "*.*")]
        )
        
        if path:
            # JPG로 저장할 경우 RGBA -> RGB 변환 처리 (에러 방지)
            if path.lower().endswith(('.jpg', '.jpeg')):
                # 투명한 배경을 사용자가 설정한 배경색으로 채움
                bg_color = self.bg_color.get()
                final_rgb = Image.new("RGB", self.final_mockup.size, bg_color)
                final_rgb.paste(self.final_mockup, (0, 0), self.final_mockup)
                final_rgb.save(path, "JPEG", quality=95)
            else:
                # PNG 등 투명도를 지원하는 포맷은 그대로 저장
                self.final_mockup.save(path)
            print(f"Successfully saved to: {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpertsMockupGenerator(root)
    root.mainloop()