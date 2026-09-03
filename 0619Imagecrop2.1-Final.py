from PIL import Image, ImageDraw
import os

def find_background_color(image):
    img_rgba = image.convert("RGBA")
    color_counts = img_rgba.getcolors(img_rgba.size[0] * img_rgba.size[1])
    
    if color_counts is None:
        return (255, 255, 255, 255)  # 색상이 너무 많으면 기본 흰색
    
    # 빈도수가 가장 높은 순으로 정렬
    sorted_colors = sorted(color_counts, key=lambda x: x[0], reverse=True)
    most_common_color = sorted_colors[0][1]
    
    # 가장 많이 차지하는 색상이 투명(alpha == 0)인 경우 -> 흰색 배경으로 강제 지정
    if most_common_color[3] == 0:
        return (255, 255, 255, 255)
        
    return most_common_color

def create_circle_mask(size):
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = size // 2
    center_x = size // 2
    center_y = size // 2
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)
    return mask

def process_circle_thumbnail(image, bg_color, thumbnail_size):
    width, height = image.size
    square_size = max(width, height)

    # 1. 감지된 배경색(또는 투명일 경우 흰색)으로 정사각형 캔버스 생성
    square_image = Image.new('RGBA', (square_size, square_size), bg_color)

    # 2. 원본 이미지를 정중앙에 붙여넣기 (투명 배경 PNG라면 흰색 배경 위에 얹어짐)
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    square_image.paste(image, (paste_x, paste_y), image if image.mode == 'RGBA' else None)

    # 3. 원형 마스크 생성
    circle_mask = create_circle_mask(square_size)

    # 4. 원형 마스크를 적용하여 원 안쪽만 남기고 바깥쪽은 완전 투명하게 처리
    result = Image.new('RGBA', (square_size, square_size), (0, 0, 0, 0))
    result.paste(square_image, (0, 0), mask=circle_mask)

    # 5. 최종 리사이즈 (LANCZOS로 쨍한 색감 유지)
    final_image = result.resize(thumbnail_size, resample=Image.LANCZOS)
    return final_image

def create_circle_thumbnail(input_folder_path, output_folder_path, thumbnail_size):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            image = Image.open(os.path.join(input_folder_path, filename)).convert("RGBA")

            # 1. 배경색 추출 (투명 배경이 최다일 경우 자동으로 흰색으로 세팅됨)
            bg_color = find_background_color(image)

            # 2. 정사각형 정렬, 배경 채우기, 원형 크롭 및 바깥쪽 투명화 처리
            final_image = process_circle_thumbnail(image, bg_color, thumbnail_size)

            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder_path, output_filename)
            final_image.save(output_path, "PNG")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Usage example
input_folder_path = "./thumb_origin"
output_folder_path = "./thumb_cropped_resized"
thumbnail_size = (142, 142)

create_circle_thumbnail(input_folder_path, output_folder_path, thumbnail_size)