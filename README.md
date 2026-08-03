# bhy

Shoulder-surfing alarm. Point a webcam at the space behind you and it warns you
when someone who should not be reading your screen is standing there.

People you have enrolled are ignored. Everyone else raises an alert. A colleague
who normally sits at their own desk is fine there but raises a warning if they
come close to the camera, which is the case worth noticing.

## How it works

Three models, all running locally on CPU. Nothing leaves the machine.

| stage | model | job |
| --- | --- | --- |
| person detection | YOLO26 Nano (`yolo26n.pt`) | draw a box round each person |
| face detection | YuNet (`face_detection_yunet_2023mar.onnx`) | find faces and 5 landmarks |
| face recognition | SFace (`face_recognition_sface_2021dec.onnx`) | turn a face into a 128-d vector |

Faces are compared to the enrolled photos by cosine similarity. The person
detector only feeds the "N person(s) detected" count; alerts are driven by faces,
so somebody facing away is boxed but not judged.

## Requirements

- Python 3.14 (developed against 3.14.6)
- A webcam
- Windows for window pinning; everything else is cross-platform

```
pip install opencv-python ultralytics
```

Developed with opencv-python 5.0.0, ultralytics 8.4.115, numpy 2.5.1, torch 2.13.0+cpu.
OpenCV must be a build with a HighGUI backend (the pip wheel is fine). The UI uses
`WIN32UI`, not Qt, so it draws its own buttons rather than using `cv2.createButton`.

## Setup

`models/` and `images/` are gitignored, so a fresh clone needs both filled in.

### 1. Models

The two ONNX files come from the OpenCV model zoo. Use the `media.` host: the
`raw.githubusercontent.com` URLs return Git LFS pointer files, not the weights.

```sh
mkdir -p models
base=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  $base/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  $base/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Expect roughly 228 KB and 37 MB. A few hundred bytes of ASCII means you got the
LFS pointer instead. `yolo26n.pt` (5.3 MB) is downloaded by ultralytics on first
run if it is missing.

### 2. Enrol yourself

Put a photo of your face in `images/owner/`. Frontal, well lit, face reasonably
large in the frame. This is the single most important file: without it you set off
your own alarm.

## Trust tiers

A photo's folder decides how it is treated, so filenames are free-form. Every face
found in a photo is enrolled, so group shots work.

| folder | meaning | when it alerts |
| --- | --- | --- |
| `images/owner/` | you | never, at any distance |
| `images/colleagues/` | trusted at their desk | **warning** when close to the camera |
| `images/ignored/` | never flag, matched loosely | never |

Empty folders are fine and are reported as unused at startup.

### Ignore zones

`images/ignored/*.txt` suppress alerts by **location** instead of identity: any
face whose centre falls inside a box is ignored, whoever it is. Good for a
colleague at a fixed desk in the background, and much more reliable than
recognising a face the camera only ever sees in profile.

One zone per line, `x1 y1 x2 y2`:

```
# a desk in the upper right quarter of the view
0.60 0.10 0.95 0.45
```

- Values within 0..1 are fractions of the frame. Prefer these: they survive a
  camera resolution change.
- Anything larger is read as pixels, and will not follow a resolution change.
- Corners in any order, commas allowed, `#` comments and blank lines ignored.
- Malformed lines are reported with a line number and skipped.

To add one without doing arithmetic: run the app, press `r`, drag a box over the
video. It is appended to `images/ignored/regions.txt` as fractions and reloaded on
the next start. Zones are outlined in grey on the video.

Zones are deliberately identity-blind, which means a stranger standing inside one
is also ignored. Keep them tight around the area you mean.

## Running

```sh
python bhy.py
```

Or add `cli/` to `PATH` and run `bhy` from anywhere. `cli/bhy.cmd` resolves the
project from its own location and prefers `.venv\Scripts\python.exe`, falling back
to whatever `python` is on `PATH`.

On Windows, appending to `PATH` is worth doing carefully: `setx` truncates at 1024
characters, and rewriting `HKCU\Environment\Path` with the wrong type turns
`%USERPROFILE%`-style entries into dead literals. Write it as `ExpandString` and
leave those entries unexpanded.

## Controls

The status panel sits below the video so it never covers a face.

| input | action |
| --- | --- |
| `PIN` button / `p` | always-on-top, parked in the bottom-right of the current screen |
| `HIDE` button / `h` | collapse the video, leaving just the status bar |
| `r` | zone picking on/off, then drag a box over the video |
| `q` / `Esc` / window close | quit |

Pinning follows whichever monitor the window is on and keeps clear of the taskbar.
While pinned the window returns to the corner if something moves it, so unpin
before dragging it elsewhere.

Collapsing keeps the width and drops the video, and skips the annotation drawing
while hidden. Detection keeps running, so alerts still appear in the bar.

## Status panel

| severity | headline | meaning |
| --- | --- | --- |
| green | `CLEAR` | nothing to flag |
| amber | `WARNING` | someone you know has come close |
| red | `ALERT` | a face that matches nothing enrolled |

The worst face in the frame sets the headline. Each face is labelled with its
height as a fraction of the frame, which is the number `NEAR_FACE_RATIO` is
compared against, so the panel doubles as the tuning readout.

## Tuning

All in `bhy.py`.

| constant | default | effect |
| --- | --- | --- |
| `MATCH_THRESHOLD` | `0.363` | cosine similarity for "same person". SFace's published value |
| `IGNORE_MATCH_THRESHOLD` | `0.2` | looser threshold, `images/ignored/` only |
| `NEAR_FACE_RATIO` | `0.22` | face height / frame height that counts as "close" |
| `MIN_RELIABLE_FACE_WIDTH` | `80` | below this a reference photo is flagged as noisy |
| `MAX_DISPLAY_HEIGHT` | `720` | preview is scaled down past this; detection stays full-res |
| `MIN_ZONE_PIXELS` | `12` | ignore stray click-drags when picking a zone |

`NEAR_FACE_RATIO` depends on your camera's field of view. Watch the number next to
a face at the distance you care about and set the threshold between the two cases.

Lowering `IGNORE_MATCH_THRESHOLD` starts silently ignoring strangers who merely
resemble someone enrolled. For a shoulder-surfing alarm that is a missed alert, so
prefer a better photo, or a zone, over a looser threshold.

## Reference photo quality

Recognition is only as good as the enrolled photo, and the failure is quiet.

- **Frontal.** SFace is trained on roughly frontal faces. In profile the two eyes
  nearly collapse together and the embedding stops being a stable signature: two
  separate profile photos of one person can score as low as 0.15, below the
  threshold for "same person", while also drifting closer to everyone else.
  If the camera only ever sees someone side-on, use a zone instead.
- **Large.** SFace normalises to 112x112, so a face narrower than that is upscaled
  guesswork. A ~45 px reference bottoms out near the strict threshold once the
  live view is blurred or distant.
- **Beware tight crops.** YuNet needs context around a face and will not detect one
  that fills the frame. When cropping a face out of a group photo, leave a wide
  margin, then confirm it is still detected.

Startup prints the size of every enrolled face and warns about small ones. If a
photo is not listed, it was not enrolled.

## Privacy

Everything runs locally: no network calls after the models are downloaded, and no
frames or embeddings are written to disk. `images/` and `models/` are gitignored,
so reference photos are not committed.

## Licence

GPL-3.0. See `LICENSE`.
