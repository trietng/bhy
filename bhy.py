import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# Pinning needs the screen geometry, which HighGUI does not expose. On Windows we
# ask the OS directly; elsewhere pinning still works but cannot reposition.
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
  import ctypes
  from ctypes import wintypes

  MONITOR_DEFAULTTONEAREST = 2

  class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]

# Resolve everything against the script's own folder so bhy.py runs from any
# working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

DETECTOR_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
YOLO_MODEL = os.path.join(MODEL_DIR, "yolo26n.pt")

# Faces matching a reference photo above this cosine similarity count as the same
# person. 0.363 is the reference threshold published for SFace.
MATCH_THRESHOLD = 0.363

# images/ignored only has to answer "is this the person I never want flagged", so
# it trades precision for recall. Measured on the bundled photo: the same face
# degraded (blur, dark, downscaled) bottoms out at 0.36, while a different person
# scores 0.09, so 0.20 catches the person in bad conditions with room to spare.
# Lowering it further will start silently ignoring strangers who merely resemble
# them, which for an over-the-shoulder alarm means a missed alert.
IGNORE_MATCH_THRESHOLD = 0.2

# The folder a photo sits in decides its trust level, so filenames are free-form.
# images/owner is you: never alerts, however close you are to the camera.
OWNER_DIR = os.path.join(BASE_DIR, "images", "owner")
# images/colleagues are trusted only while they stay at their own desk. The same
# face close to the camera means someone is reading over your shoulder.
COLLEAGUE_DIR = os.path.join(BASE_DIR, "images", "colleagues")
# images/ignored are never flagged at all, at any distance, matched loosely.
IGNORED_DIR = os.path.join(BASE_DIR, "images", "ignored")

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

# Zones dragged out in the app land here, and are read back on the next start.
ZONE_FILE = os.path.join(IGNORED_DIR, "regions.txt")
# Ignore a stray click that drags a couple of pixels
MIN_ZONE_PIXELS = 12

# A face taller than this fraction of the frame is near the camera. Roughly 22%
# separates arm's length from a desk a few metres away on a typical webcam; the
# on-screen number next to each face is the measured value, so it is easy to tune.
NEAR_FACE_RATIO = 0.22

# SFace normalises every face to 112x112, so a reference face narrower than this
# is upscaled and produces a noisier, less dependable feature.
MIN_RELIABLE_FACE_WIDTH = 80

# Shrink the preview if the camera hands us something taller than this.
MAX_DISPLAY_HEIGHT = 720

# Three tiers, worst one wins for the panel headline. A colleague who has come close
# is a warning; a face we cannot place at all is an alert.
OK, WARNING, ALERT = "ok", "warning", "alert"
SEVERITY_RANK = {OK: 0, WARNING: 1, ALERT: 2}

# BGR, so amber is (0, 165, 255)
SEVERITY_COLOR = {OK: (0, 200, 0), WARNING: (0, 165, 255), ALERT: (0, 0, 255)}
SEVERITY_HEADLINE = {
  OK: "CLEAR",
  WARNING: "WARNING - someone you know is close",
  ALERT: "ALERT - unrecognised person behind you",
}

# The status panel is drawn as its own strip underneath the video, not on top of
# it, so it never covers a face.
PANEL_HEIGHT = 92
PANEL_BACKGROUND = (28, 28, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# The pin button lives at the right end of the panel. Neutral greys on purpose,
# so it is never mistaken for a severity colour.
BUTTON_WIDTH = 108
BUTTON_HEIGHT = 34

def display_width(frame):
  """Width the frame will occupy once scaled, so a collapsed panel keeps it."""
  height, width = frame.shape[:2]
  if height <= MAX_DISPLAY_HEIGHT:
    return width
  return round(width * MAX_DISPLAY_HEIGHT / height)

def fit_for_display(frame):
  """Scale a frame down to MAX_DISPLAY_HEIGHT without changing its proportions."""
  if frame.shape[0] <= MAX_DISPLAY_HEIGHT:
    return frame

  size = (display_width(frame), MAX_DISPLAY_HEIGHT)
  return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def load_reference_faces(recognizer, detector, directory, kind, exclude=()):
  """Embed every face in every photo in `directory`, skipping any in `exclude`."""
  features = []
  paths = sorted(
    p for extension in IMAGE_EXTENSIONS
    for p in glob.glob(os.path.join(directory, extension))
  )

  # An empty folder is legitimate, it just means that tier is unused
  if not paths:
    print(f"{kind}: no photos in {os.path.basename(directory)}/, tier unused")

  for path in paths:
    name = os.path.basename(path)
    image = cv2.imread(path)
    if image is None:
      print(f"Warning: could not read {name}")
      continue

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
      print(f"Warning: no face found in {name}")
      continue

    # Everyone in a reference photo is trusted, largest face first.
    for face in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
      face_width, face_height = int(face[2]), int(face[3])
      feature = recognizer.feature(recognizer.alignCrop(image, face))

      # A group photo may also contain someone a stronger rule already covers,
      # so don't add a duplicate that could never be reached.
      if matches_any(recognizer, feature, exclude):
        print(f"Skipping {face_width}x{face_height} face in {name}, "
              "already covered by a stronger rule")
        continue

      note = ""
      if face_width < MIN_RELIABLE_FACE_WIDTH:
        note = " (small, matching may be unreliable)"
      features.append(feature)
      print(f"{kind}: {face_width}x{face_height} face from {name}{note}")

  return features

def matches_any(recognizer, feature, features, threshold=MATCH_THRESHOLD):
  return any(
    recognizer.match(feature, known, cv2.FaceRecognizerSF_FR_COSINE) >= threshold
    for known in features
  )

def load_ignored_zones(directory):
  """Read ignore zones from every .txt in `directory`.

  One zone per line as `x1 y1 x2 y2`. Values within 0..1 are read as fractions of
  the frame so they survive a resolution change; anything larger is read as pixels.
  Blank lines and everything after a '#' are skipped.
  """
  zones = []
  for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as handle:
      for number, line in enumerate(handle, 1):
        line = line.split("#", 1)[0].strip()
        if not line:
          continue

        parts = line.replace(",", " ").split()
        if len(parts) != 4:
          print(f"Warning: {name}:{number} needs 4 numbers, got {len(parts)}")
          continue
        try:
          x1, y1, x2, y2 = (float(part) for part in parts)
        except ValueError:
          print(f"Warning: {name}:{number} is not numeric, skipped")
          continue

        # Accept the corners in any order
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if left == right or top == bottom:
          print(f"Warning: {name}:{number} has zero area, skipped")
          continue

        fractional = max(left, top, right, bottom) <= 1.0
        zones.append((left, top, right, bottom, fractional))
        units = "fractions" if fractional else "pixels"
        print(f"Ignore zone from {name}:{number} ({units}): "
              f"{left:g},{top:g} to {right:g},{bottom:g}")

  if not zones:
    print(f"Ignore zones: none defined in {os.path.basename(directory)}/*.txt")
  return zones

def zone_to_pixels(zone, width, height):
  """Resolve a stored zone against the frame it is being tested against."""
  left, top, right, bottom, fractional = zone
  if fractional:
    return (round(left * width), round(top * height),
            round(right * width), round(bottom * height))
  return (round(left), round(top), round(right), round(bottom))

def in_any_zone(point, zones, width, height):
  x, y = point
  for zone in zones:
    left, top, right, bottom = zone_to_pixels(zone, width, height)
    if left <= x <= right and top <= y <= bottom:
      return True
  return False

def draw_zones(frame, zones):
  """Outline the zones so it is obvious where alerts are suppressed."""
  for zone in zones:
    left, top, right, bottom = zone_to_pixels(zone, frame.shape[1], frame.shape[0])
    cv2.rectangle(frame, (left, top), (right, bottom), (150, 150, 150), 1)
    cv2.putText(frame, "ignore zone", (left + 4, max(top - 6, 12)),
                FONT, 0.45, (150, 150, 150), 1)

def classify_face(recognizer, feature, face, frame_width, frame_height,
                  owners, colleagues, ignored, zones):
  """Return (label, severity) for one detected face."""
  # You are trusted unconditionally, distance is irrelevant.
  if matches_any(recognizer, feature, owners):
    return "you", OK

  # A zone suppresses whoever is standing in it, identity is not consulted
  x, y, w, h = face[:4]
  if in_any_zone((x + w / 2, y + h / 2), zones, frame_width, frame_height):
    return "ignored (zone)", OK

  # Checked before the distance rule: "never flag them" outranks "only at a desk".
  if matches_any(recognizer, feature, ignored, IGNORE_MATCH_THRESHOLD):
    return "ignored", OK

  near = face[3] / frame_height >= NEAR_FACE_RATIO

  if matches_any(recognizer, feature, colleagues):
    if near:
      return "colleague, close", WARNING
    return "colleague, at desk", OK

  return "unknown", ALERT

def anchor_bottom_right(window):
  """Park the window in the bottom-right of the screen it currently sits on."""
  if not IS_WINDOWS:
    return

  user32 = ctypes.windll.user32
  hwnd = user32.FindWindowW(None, window)
  if not hwnd:
    return

  info = MONITORINFO()
  info.cbSize = ctypes.sizeof(MONITORINFO)
  monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
  if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
    return

  # rcWork excludes the taskbar, so the status panel stays clickable
  rect = wintypes.RECT()
  user32.GetWindowRect(hwnd, ctypes.byref(rect))
  target = (info.rcWork.right - (rect.right - rect.left),
            info.rcWork.bottom - (rect.bottom - rect.top))

  # Only move when it has drifted, otherwise we fight the compositor every frame
  if (rect.left, rect.top) != target:
    cv2.moveWindow(window, *target)

def apply_pin(window, pinned):
  cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1 if pinned else 0)
  if pinned:
    anchor_bottom_right(window)

def append_zone(path, box, width, height):
  """Store a dragged box as fractions of the frame and return it as a zone."""
  left, right = sorted((box[0] / width, box[2] / width))
  top, bottom = sorted((box[1] / height, box[3] / height))
  # A drag can end outside the video, so keep it on the frame
  left, top, right, bottom = (min(max(value, 0.0), 1.0)
                              for value in (left, top, right, bottom))

  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "a", encoding="utf-8") as handle:
    handle.write(f"{left:.4f} {top:.4f} {right:.4f} {bottom:.4f}\n")

  return (left, top, right, bottom, True)

def on_mouse(event, x, y, flags, state):
  """HighGUI has no widgets, so each button is a rectangle we hit-test."""
  if event == cv2.EVENT_LBUTTONDOWN:
    for name, (left, top, right, bottom) in state["buttons"].items():
      if left <= x <= right and top <= y <= bottom:
        state["clicked"] = name
        return

  if not state["picking"]:
    return

  # While picking, a drag over the video defines a zone
  if event == cv2.EVENT_LBUTTONDOWN:
    state["drag"] = [x, y, x, y]
  elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
    state["drag"][2:] = [x, y]
  elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
    state["drag"][2:] = [x, y]
    state["drag_done"] = True

def fitted_scale(text, max_width, scale, thickness):
  """Shrink `scale` just enough that `text` fits inside `max_width`."""
  (text_width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
  if text_width <= max_width:
    return scale
  return scale * max_width / text_width

def draw_button(panel, rect, label, active):
  """Draw the pin button, filled while active."""
  left, top, right, bottom = rect
  if active:
    cv2.rectangle(panel, (left, top), (right, bottom), (95, 95, 95), -1)
    border, text_color = (240, 240, 240), (255, 255, 255)
  else:
    border, text_color = (110, 110, 110), (200, 200, 200)

  cv2.rectangle(panel, (left, top), (right, bottom), border, 1)

  scale = fitted_scale(label, right - left - 14, 0.5, 1)
  (text_width, text_height), _ = cv2.getTextSize(label, FONT, scale, 1)
  origin = (left + (right - left - text_width) // 2,
            top + (bottom - top + text_height) // 2)
  cv2.putText(panel, label, origin, FONT, scale, text_color, 1)

def draw_status_panel(width, severity, concerns, people, pinned, collapsed):
  """Build the strip that sits below the video. Returns (panel, {name: rect})."""
  panel = np.full((PANEL_HEIGHT, width, 3), PANEL_BACKGROUND, np.uint8)
  color = SEVERITY_COLOR[severity]

  # A colour bar along the top edge reads at a glance, even out of focus
  cv2.rectangle(panel, (0, 0), (width, 5), color, -1)

  # Buttons sit right to left: pin on the end, collapse beside it
  margin, gap = 16, 8
  top = (PANEL_HEIGHT - BUTTON_HEIGHT) // 2 + 3
  bottom = top + BUTTON_HEIGHT
  pin_left = width - margin - BUTTON_WIDTH
  collapse_left = pin_left - gap - BUTTON_WIDTH

  buttons = {
    "pin": (pin_left, top, width - margin, bottom),
    "collapse": (collapse_left, top, collapse_left + BUTTON_WIDTH, bottom),
  }
  draw_button(panel, buttons["pin"], "UNPIN" if pinned else "PIN", pinned)
  draw_button(panel, buttons["collapse"], "SHOW" if collapsed else "HIDE", collapsed)

  # Text stops short of the buttons so the two can never overlap
  usable = collapse_left - margin - 12

  headline = SEVERITY_HEADLINE[severity]
  cv2.putText(panel, headline, (margin, 44), FONT,
              fitted_scale(headline, usable, 0.8, 2), color, 2)

  detail = " | ".join(concerns) if concerns else "nothing to flag"
  detail = f"{detail}   -   {people} person(s) detected"
  cv2.putText(panel, detail, (margin, 74), FONT,
              fitted_scale(detail, usable, 0.5, 1), (190, 190, 190), 1)

  return panel, buttons

def main():
  for path in (DETECTOR_MODEL, RECOGNIZER_MODEL):
    if not os.path.exists(path):
      print(f"Error: missing model file {path}")
      return

  # Load the YOLO26 Nano model (auto-downloads if not present)
  # The Nano model is optimized for CPU/Edge environments
  model = YOLO(YOLO_MODEL)

  # YuNet detects faces, SFace turns each one into a comparable 128-d feature
  detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320), 0.6, 0.3, 5000)
  recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

  owners = load_reference_faces(recognizer, detector, OWNER_DIR, "Always trusted")
  ignored = load_reference_faces(recognizer, detector, IGNORED_DIR, "Always ignored",
                                 exclude=owners)
  colleagues = load_reference_faces(recognizer, detector, COLLEAGUE_DIR,
                                    "Trusted at a distance", exclude=owners + ignored)

  zones = load_ignored_zones(IGNORED_DIR)

  if not owners:
    print(f"Warning: no photo of you in {OWNER_DIR}, you will trigger alerts yourself")
  if not owners and not colleagues and not ignored and not zones:
    print("Warning: nothing configured, every face will be treated as unknown")

  # Open the default webcam
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("Error: Could not open video stream.")
    return

  # AUTOSIZE keeps the window at the frame's own size. A WINDOW_NORMAL window
  # rescales the image to the window shape, which distorts the aspect ratio.
  window = "bhy"
  cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

  # "buttons" holds each hit box in full-image coordinates, refreshed every frame
  # because they depend on the current frame width and on whether we are collapsed.
  state = {"pinned": False, "collapsed": False, "clicked": None, "buttons": {},
           "picking": False, "drag": None, "drag_done": False}
  cv2.setMouseCallback(window, on_mouse, state)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Run YOLO26 inference
    results = model.predict(frame, classes=0, conf=0.5, verbose=False)

    # Draw the results on the frame
    # The ultralytics API provides a convenient plot() method. Collapsed means
    # nobody is looking at the video, so skip the drawing but keep detecting.
    annotated_frame = None if state["collapsed"] else results[0].plot()
    if annotated_frame is not None:
      draw_zones(annotated_frame, zones)

    # Recognise every face in the frame and flag the ones we don't know
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    severity = OK
    concerns = []
    for face in faces if faces is not None else []:
      feature = recognizer.feature(recognizer.alignCrop(frame, face))
      label, face_severity = classify_face(recognizer, feature, face, width, height,
                                           owners, colleagues, ignored, zones)

      # The worst face in the frame decides the overall status
      if SEVERITY_RANK[face_severity] > SEVERITY_RANK[severity]:
        severity = face_severity

      x, y, w, h = face[:4].astype(int)
      ratio = h / height
      if face_severity != OK:
        concerns.append(f"{label} {ratio:.2f}")

      if annotated_frame is not None:
        color = SEVERITY_COLOR[face_severity]
        # The ratio is what NEAR_FACE_RATIO is compared against, so show it to tune
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_frame, f"{label} {ratio:.2f}", (x, max(y - 8, 12)),
                    FONT, 0.6, color, 2)

    # Status lives in its own strip below the video so it never hides a face.
    # Collapsed keeps the same width, so only the height changes.
    display = None if annotated_frame is None else fit_for_display(annotated_frame)

    # The drag is in display coordinates, so it is drawn after scaling
    if display is not None and state["picking"]:
      cv2.putText(display, "zone picking: drag a box, 'r' to stop", (10, 22),
                  FONT, 0.5, (255, 255, 0), 1)
      if state["drag"]:
        left, top, right, bottom = state["drag"]
        cv2.rectangle(display, (left, top), (right, bottom), (255, 255, 0), 1)

    panel_width = display_width(frame) if display is None else display.shape[1]
    panel, buttons = draw_status_panel(panel_width, severity, concerns,
                                       len(results[0].boxes), state["pinned"],
                                       state["collapsed"])

    # The panel is stacked under the video, so shift the hit boxes down to match
    top_offset = 0 if display is None else display.shape[0]
    state["buttons"] = {
      name: (left, top + top_offset, right, bottom + top_offset)
      for name, (left, top, right, bottom) in buttons.items()
    }

    cv2.imshow(window, panel if display is None else cv2.vconcat([display, panel]))

    # Quit on 'q'/ESC, or when the window's close button is pressed
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
      break
    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
      break

    # A finished drag becomes a zone, saved so it survives a restart
    if state["drag_done"]:
      state["drag_done"] = False
      box, state["drag"] = state["drag"], None
      if display is None:
        pass
      elif (abs(box[2] - box[0]) < MIN_ZONE_PIXELS
            or abs(box[3] - box[1]) < MIN_ZONE_PIXELS):
        print("Zone too small, not saved")
      else:
        zone = append_zone(ZONE_FILE, box, display.shape[1], display.shape[0])
        zones.append(zone)
        print(f"Saved zone to {os.path.basename(ZONE_FILE)}: "
              f"{zone[0]:.4f} {zone[1]:.4f} {zone[2]:.4f} {zone[3]:.4f}")

    # Keyboard equivalents of the two buttons, plus zone picking
    if key == ord("p"):
      state["clicked"] = "pin"
    elif key == ord("h"):
      state["clicked"] = "collapse"
    elif key == ord("r"):
      state["picking"] = not state["picking"]
      state["drag"] = None
      if state["picking"]:
        # Aiming needs the video, so expand if it is hidden
        state["collapsed"] = False
      print("Zone picking " + ("on: drag a box over the video" if state["picking"]
                               else "off"))

    action = state["clicked"]
    state["clicked"] = None

    if action == "pin":
      state["pinned"] = not state["pinned"]
      apply_pin(window, state["pinned"])
    else:
      if action == "collapse":
        state["collapsed"] = not state["collapsed"]
      if state["pinned"]:
        # Collapsing changes the window height, so the corner has to be recomputed.
        # This also re-anchors if something moved the window.
        anchor_bottom_right(window)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
