import os
import re
import base64
import cv2
import numpy as np
from google import genai
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from gtts import gTTS

from config import Config
from models import db, User, Design, Booking
from dotenv import load_dotenv

from ultralytics import YOLO
load_dotenv()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

yolo_model = YOLO("yolov8n-seg.pt")

# -----------------------------
# IMAGE LOADER
# -----------------------------
def load_image_cv(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

# =====================================================
# ROOM ANALYZER
# =====================================================

def validate_room_image(image_path):

    img = load_image_cv(image_path)
    if img is None:
        return False

    h, w, _ = img.shape
    if h < 200 or w < 200:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur check
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
        return False

    # indoor structure check
    edges = cv2.Canny(gray,80,150)
    edge_ratio = np.sum(edges > 0) / (h * w)

    if edge_ratio < 0.01:
        return False

    lines = cv2.HoughLinesP(
        edges,1,np.pi/180,80,
        minLineLength=100,maxLineGap=20
    )

    if lines is None:
        return False

    return True


def estimate_dimensions(img):

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,60,140)

    density = np.sum(edges > 0)/(h*w)

    if density < 0.03:
        width_m, depth_m = 6.0, 4.5
    elif density < 0.06:
        width_m, depth_m = 4.5, 4.0
    else:
        width_m, depth_m = 3.5, 3.0

    area = round(width_m*depth_m,1)

    return {
        "width": f"{width_m} m (approx)",
        "depth": f"{depth_m} m (approx)",
        "area": f"{area} sqm"
    }


def detect_structural_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,70,150)

    features = ["walls","floor"]

    if np.mean(edges[:int(img.shape[0]*0.2), :]) > 10:
        features.append("ceiling")

    lines = cv2.HoughLinesP(
        edges,1,np.pi/180,80,
        minLineLength=80,maxLineGap=10
    )

    if lines is not None:
        features.append("windows/doors")

    return features


def analyze_room(image_path):

    img = load_image_cv(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    if brightness > 170:
        lighting = "bright"
    elif brightness > 100:
        lighting = "moderate"
    else:
        lighting = "dim"

    edges = cv2.Canny(gray,60,140)
    density = np.sum(edges > 0)/(img.shape[0]*img.shape[1])

    # ONLY empty or furnished
    if density < 0.03:
        state = "Empty"
        free_space = "high"
    elif density < 0.06:
        state = "Furnished"
        free_space = "medium"
    else:
        state = "Furnished"
        free_space = "low"

    return {
        "room_state": state,
        "elements": detect_structural_features(img),
        "free_space": free_space,
        "lighting": lighting,
        "dimensions": estimate_dimensions(img)
    }


# =====================================================
# EVERYTHING BELOW IS ORIGINAL (UNCHANGED)
# =====================================================

# -----------------------------
# AI DESIGN SCORE
# -----------------------------
def calculate_design_score(style, room_type):
    score = 70

    if style == "Modern":
        score += 10
    elif style == "Classic":
        score += 8
    else:
        score += 6

    if room_type == "Living Room":
        score += 10
    else:
        score += 5

    return min(score, 100)


# -----------------------------
# FURNITURE RECOMMENDER
# -----------------------------
def recommend_furniture(style):
    if style == "Modern":
        return [
            {"name": "Minimal Sofa", "price": "₹18,000"},
            {"name": "Glass Coffee Table", "price": "₹6,500"},
            {"name": "LED Floor Lamp", "price": "₹3,200"}
        ]
    elif style == "Classic":
        return [
            {"name": "Wooden Sofa Set", "price": "₹22,000"},
            {"name": "Antique Table", "price": "₹8,000"},
            {"name": "Classic Wall Lamp", "price": "₹4,200"}
        ]
    else:
        return [
            {"name": "Simple Fabric Sofa", "price": "₹15,000"},
            {"name": "Compact Table", "price": "₹5,000"},
            {"name": "Soft Light Lamp", "price": "₹2,800"}
        ]


# -----------------------------
# MULTILINGUAL VOICE ASSISTANT
# -----------------------------
def generate_voice(text, lang="en"):
    os.makedirs("static/audio", exist_ok=True)
    tts = gTTS(text=text, lang=lang)
    audio_path = "static/audio/output.mp3"
    tts.save(audio_path)
    return audio_path


# -----------------------------
# AI FURNITURE OPTIMIZER
# -----------------------------
client = genai.Client(api_key="YOUR_NEW_KEY_HERE")

def ai_furniture_optimizer(room_type, room_size, furniture, image_path=None):

    prompt = f"""
You are an expert interior designer.

Room Type: {room_type}
Room Size: {room_size}
Furniture Items: {furniture}

Analyze the room image if provided and suggest optimal furniture placement.

Return 5 clear bullet points.
"""

    contents = [prompt]

    if image_path:
        with open(image_path, "rb") as img:
            contents.append({
                "mime_type": "image/jpeg",
                "data": img.read()
            })

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )
        return response.text

    except Exception:
        return """• Place large furniture against walls.
• Keep center space clear.
• Use corners for storage.
• Avoid blocking windows.
• Create a focal point for balance."""


# -----------------------------
# DRAW FURNITURE ZONES
# -----------------------------
from PIL import ImageDraw

def draw_layout_zones(image_path):

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    width, height = img.size

    zones = [
        ("SOFA ZONE", (50, height-220, width//2, height-50), "blue"),
        ("TV ZONE", (width//2+50, 50, width-50, 200), "red"),
        ("WALKING SPACE", (width//3, height//3, width*2//3, height*2//3), "green")
    ]

    for label, box, color in zones:
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0]+10, box[1]+10), label, fill=color)

    os.makedirs("static/optimized", exist_ok=True)
    output_path = "static/optimized/layout_result.png"
    img.save(output_path)

    return output_path


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# REGISTER
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            return render_template("register.html", result="⚠️ Invalid email format.")

        if len(password) < 8 or not re.search(r"\d", password):
            return render_template("register.html", result="⚠️ Password must be 8+ chars with numbers.")

        if User.query.filter_by(email=email).first():
            return render_template("register.html", result="⚠️ Email already registered.")

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")


# -----------------------------
# LOGIN
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect("/dashboard")

        return render_template("login.html", result="⚠️ Invalid email or password.")

    return render_template("login.html")


# -----------------------------
# DASHBOARD
# ----------------------------


# -----------------------------
# DESIGN PAGE (ROOM ANALYZER)
# -----------------------------
@app.route("/design", methods=["GET","POST"])
def design():

    if "user_id" not in session:
        return redirect("/login")

    if request.method=="POST":

        image_file = request.files.get("room_image")
        if not image_file:
            return render_template("design.html",
                result="Invalid image: Please upload a clear photo of an indoor room.")

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        filename = str(datetime.utcnow().timestamp())+"_"+image_file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(filepath)

        if not validate_room_image(filepath):
            return render_template("design.html",
                result="Invalid image: Please upload a clear photo of an indoor room.")

        analysis = analyze_room(filepath)

        result_text = f"""
Room Validity: Valid
Room State: {analysis['room_state']}

Estimated Dimensions:
- Width: {analysis['dimensions']['width']}
- Depth: {analysis['dimensions']['depth']}
- Area: {analysis['dimensions']['area']}

Detected Elements:
- {'\n- '.join(analysis['elements'])}

Free Space: {analysis['free_space']}
Lighting: {analysis['lighting']}
"""

        return render_template(
            "design.html",
            result=result_text,
            uploaded_image=filepath
        )

    return render_template("design.html")


# -----------------------------
# FURNITURE OPTIMIZER
# -----------------------------
@app.route("/furniture_optimizer", methods=["GET","POST"])
def furniture_optimizer():

    if "user_id" not in session:
        return redirect("/login")

    suggestions = []
    optimized_image = None

    if request.method == "POST":

        room_type = request.form.get("room_type")
        room_size = request.form.get("room_size")
        furniture = request.form.get("furniture")

        image_file = request.files.get("room_image")
        image_path = None

        if image_file and image_file.filename:
            os.makedirs("static/uploads", exist_ok=True)
            image_path = os.path.join("static/uploads", image_file.filename)
            image_file.save(image_path)

        ai_result = ai_furniture_optimizer(
            room_type,
            room_size,
            furniture,
            image_path
        )

        suggestions = [s for s in ai_result.split("\n") if s.strip()]

        if image_path:
            optimized_image = draw_layout_zones(image_path)

    return render_template(
        "furniture_optimizer.html",
        suggestions=suggestions,
        optimized_image=optimized_image
    )

@app.route("/style", methods=["GET","POST"])
def style():

    print("🔥 STYLE ROUTE EXECUTED")

    suggestions = []

    if request.method == "POST":
        style_choice = request.form.get("style")

        data = {

            "Modern": [
                "Use white or light grey marble flooring for a clean premium look.",
                "Choose neutral wall colors like white, grey, or beige.",
                "Use sleek modular furniture with straight lines.",
                "Install smart LED strip lighting or recessed ceiling lights.",
                "Add glass or metal accent tables.",
                "Use minimal wall decor with abstract art.",
                "Keep windows large with light curtains for openness.",
                "Use monochrome or dual-tone color palettes.",
                "Add indoor plants in simple geometric pots.",
                "Avoid heavy textures to maintain openness."
            ],

            "Minimal": [
                "Use plain matte wall colors like off-white or soft grey.",
                "Choose light wooden flooring or simple tiles.",
                "Limit furniture to only essential pieces.",
                "Use hidden storage to keep surfaces clutter-free.",
                "Prefer soft natural lighting and plain curtains.",
                "Keep decoration minimal — one or two statement pieces only.",
                "Use low-profile furniture with clean shapes.",
                "Maintain open floor space for a calm feel.",
                "Use neutral cushions and simple fabrics.",
                "Avoid busy patterns or multiple textures."
            ],

            "Traditional": [
                "Use warm wall colors like cream, mustard, or earthy tones.",
                "Choose polished wooden flooring or classic tiles.",
                "Add wooden furniture with carved details.",
                "Use warm yellow lighting or chandeliers.",
                "Decorate walls with traditional artwork or frames.",
                "Add heavy curtains with rich textures.",
                "Use patterned rugs for a cozy feel.",
                "Include brass or antique decorative pieces.",
                "Use wooden or stone center tables.",
                "Layer textures for a rich cultural look."
            ],

            "Luxury": [
                "Use premium marble flooring with glossy finish.",
                "Choose deep wall tones like navy, charcoal, or warm beige.",
                "Install designer lighting or chandeliers.",
                "Use velvet or leather upholstered furniture.",
                "Add gold or metallic accents in decor.",
                "Use large statement mirrors to enhance space.",
                "Layer lighting with ceiling + floor lamps.",
                "Add premium curtains reaching floor level.",
                "Use textured wall panels or wallpapers.",
                "Keep layout spacious with fewer but high-end pieces."
            ],

            "Bohemian": [
                "Use warm earthy wall colors like terracotta or sandy beige.",
                "Mix patterned rugs and colorful textiles.",
                "Use natural materials like rattan and bamboo.",
                "Add lots of indoor plants and hanging planters.",
                "Use mixed-color cushions and throws.",
                "Decorate with handmade or artistic wall hangings.",
                "Use soft warm lighting with lantern-style lamps.",
                "Layer different textures for a relaxed vibe.",
                "Add vintage or handcrafted furniture pieces.",
                "Mix colors freely without strict symmetry."
            ]
        }

        suggestions = data.get(style_choice, [])

    return render_template(
        "style.html",
        style_suggestions=suggestions
    )

# -----------------------------
# 3D DEMO PAGE
# -----------------------------
@app.route("/demo3d")
def demo3d():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("demo3d.html")

# -----------------------------
# BUDGET PLANNER PAGE
# -----------------------------
@app.route("/budget", methods=["GET","POST"])
def budget():

    if "user_id" not in session:
        return redirect("/login")

    budget_result = None

    if request.method == "POST":
        total_budget = int(request.form.get("total_budget"))

        budget_result = {
            "Furniture": int(total_budget * 0.40),
            "Lighting": int(total_budget * 0.15),
            "Wall & Paint": int(total_budget * 0.20),
            "Decor": int(total_budget * 0.15),
            "Miscellaneous": int(total_budget * 0.10)
        }

    return render_template(
        "budget.html",
        budget_result=budget_result
    )


@app.route("/ar_camera")
def ar_camera():
    return render_template("ar_camera.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/catalog")
@login_required

def catalog():
    return render_template("catalog.html")

@app.route("/bookings")
@login_required
def bookings():
    return render_template("bookings.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect("/")



# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
