import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
import tempfile
import labelme.utils
from labelme import QT5
from labelme.shape import Shape, MultipoinstShape, MaskShape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.
use_video = True
sam2_checkpoint = "/code_projects/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
assert os.path.exists(sam2_checkpoint), f"{sam2_checkpoint} does not exist! Please download sam2 checkpoint first."
model_cfg = "sam2_hiera_t.yaml"
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2, build_sam2_video_predictor

use_sam = True
CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


def init_sam():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    MOVE_SPEED = 5.0

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    return sam2_model


# image
# predictor = SAM2ImagePredictor(sam2_model)
# video
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white",
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white",
               linewidth=1.25)


# def sam2():

class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "polygonSAM": False,
            },
        )
        self.sam_config = kwargs.pop(
            "sam",
            {
                "maxside": 2048,
                "approxpoly_epsilon": 0.5,
                # "weights": "vit-h",
                # "weights": "vit_l",
                "device": "cuda"
            }
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current_prompts = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.sam_predictor: SAM2VideoPredictor | SAM2ImagePredictor = None
        self.sam_mask = MaskShape()
        self.hided_sam_mask = None  # to hide mask temporarily
        self.sam_video_frame_idx = None
        self.sam_video_init = False
        self.points = defaultdict(list)
        self.sam_propagated_video = False

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "polygonSAM",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value
        self.sam_mask = MaskShape()
        self.current_prompts = None

    def loadSamPredictor(self):
        if not use_sam:
            return
        if not self.sam_predictor:
            if not use_video:
                sam2_model = init_sam()
                predictor = SAM2ImagePredictor(sam2_model)
                self.sam_predictor = predictor

                # old sam
                # import torch
                # from segment_anything import sam_model_registry, SamPredictor
                # cachedir = appdirs.user_cache_dir("labelme")
                # os.makedirs(cachedir, exist_ok=True)
                # weight_file = os.path.join(cachedir, self.sam_config["weights"] + ".pth")
                # weight_urls = {
                #     "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                #     "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                #     "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                # }
                # if not os.path.isfile(weight_file):
                #     torch.hub.download_url_to_file(weight_urls[self.sam_config["weights"]], weight_file)
                # sam = sam_model_registry[self.sam_config["weights"]](checkpoint=weight_file)
                # if self.sam_config["device"] == "cuda" and torch.cuda.is_available():
                #     sam.to(device="cuda")
                # self.sam_predictor = SamPredictor(sam)
            else:
                try:
                    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

                    self.sam_predictor = predictor
                except Exception as e:
                    print(e)
        # self.samEmbedding()

    def reset_video_state(self):
        print("reset_video_state")
        self.sam_video_init = False
        self.sam_video_frame_idx = None
        self.sam_propagated_video = False
        self.video_masks = None
        self.sam_mask = MaskShape()
        self.current_prompts = None
        self.repaint()

    def update_frame_idx(self, idx):
        self.sam_video_frame_idx = idx

    def sam_hide(self):
        if self.sam_mask is not None and self.hided_sam_mask is None:
            self.hided_sam_mask = self.sam_mask.copy()
            self.sam_mask = MaskShape()
            self.sam_mask.paint(self._painter)
            self.repaint()

    def sam_unhide(self):
        if self.sam_mask is not None:
            self.sam_mask = self.hided_sam_mask
            self.sam_mask.paint(self._painter)
            self.repaint()
            self.hided_sam_mask = None

    def update_sam(self):
        if not use_sam:
            return
        print("update_sam")

        image = self.pixmap.toImage().copy()
        img_size = image.size()
        s = image.bits().asstring(img_size.height() * img_size.width() * image.depth() // 8)
        image = np.frombuffer(s, dtype=np.uint8).reshape([img_size.height(), img_size.width(), image.depth() // 8])
        image = image[:, :, :3].copy()
        h, w, _ = image.shape
        self.sam_image_scale = self.sam_config["maxside"] / max(h, w)
        self.sam_image_scale = min(1, self.sam_image_scale)
        print("self.sam_image_scale", self.sam_image_scale)
        if not use_video:
            mage = cv2.resize(image, None, fx=self.sam_image_scale, fy=self.sam_image_scale,
                              interpolation=cv2.INTER_LINEAR)
            self.sam_predictor.set_image(image)
        elif not self.sam_video_init:
            self.max_video_num_frames = 32

            self.jpg_dir = tempfile.TemporaryDirectory().name
            print("jpg_dir", self.jpg_dir)
            os.makedirs(self.jpg_dir, exist_ok=True)

            start_i = None
            img_folder_files = sorted(os.listdir(self.img_dir))
            count_imgs = 0
            for i, filename in enumerate(img_folder_files):
                if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
                    count_imgs += 1
            # find current file
            for i, filename in enumerate(img_folder_files):
                if not self.current_image_path.endswith(filename):
                    continue
                start_i = i
                break
            self.sam_video_frame_idx = 0
            images_found = 0
            i = 0
            # bit hacky for now. The sam2 code accepts a folder with images. So we create a temporary folder.
            while images_found < min(count_imgs, self.max_video_num_frames) and start_i + i < len(img_folder_files):
                filename = img_folder_files[start_i + i]
                i += 1

                if filename.endswith(".png") or filename.endswith(".jpg"):
                    # Open the image
                    png_image = Image.open(os.path.join(self.img_dir, filename))

                    rgb_image = png_image.convert("L")

                    new_filename = f"{images_found:05d}.jpg"
                    # Save the image in JPG format
                    rgb_image.save(os.path.join(self.jpg_dir, new_filename), "JPEG")  # save
                    images_found += 1
            print("Conversion complete.")
            self.num_tracking_video_frames = images_found

            self.video_inference_state = self.sam_predictor.init_state(video_path=self.jpg_dir)
            self.sam_predictor.reset_state(self.video_inference_state)
            self.sam_video_init = True
        elif self.sam_propagated_video:
            print("self.video_masks", len(self.video_masks))
            print("self.sam_video_frame_idx", self.sam_video_frame_idx)
            mask = self.video_masks[self.sam_video_frame_idx]
            self.sam_mask.setScaleMask(self.sam_image_scale, mask.squeeze(0))

        self.update()

    def samPrompt(self, points, labels, click=True):
        if not use_video:
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points * self.sam_image_scale,
                point_labels=labels,
                mask_input=self.sam_mask.logits[None, :, :] if self.sam_mask.logits is not None else None,
                multimask_output=False
            )
            self.sam_mask.logits = logits[np.argmax(scores), :, :]
            mask = self.sam_mask.logits.copy()
            print("image mask", mask.shape)
            self.sam_mask.setScaleMask(self.sam_image_scale, mask)
        elif (not self.sam_propagated_video or click) and self.sam_video_init:
            # Currently only prompt on click or on the first prompt point. To avoid hovering and adding points
            print("points", points, labels)
            with np.printoptions(precision=2, suppress=True):
                print("points", points)
                # print('  self.points[self.sam_video_frame_idx] ',   self.points[self.sam_video_frame_idx] )
                print("self.sam_video_frame_idx", self.sam_video_frame_idx)
            frame_idx, out_obj_ids, out_mask_logits = self.sam_predictor.add_new_points(
                inference_state=self.video_inference_state,
                frame_idx=self.sam_video_frame_idx,  # ann_frame_idx
                obj_id=1,  # todo not sure what to do with this. I guess its the object that we want to follow
                points=points,
                labels=labels,
                # clear_old_points=False,
            )
            print("out_mask_logits[0]", out_mask_logits[0].max(), out_mask_logits[0].mean(),
                  out_mask_logits[0].median())
            masks = (out_mask_logits[0] > 0.0).cpu().numpy()
            # print('video mask', mask.shape)
            print("len self.video_masks", len(masks))
            # mask = masks[np.argmax(scores), :, :]
            self.sam_mask.setScaleMask(self.sam_image_scale, masks[0])

    def sam_propagate_video(self):
        if self.sam_predictor is None or not self.sam_video_init:
            return None

        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_predictor.propagate_in_video(
                self.video_inference_state, max_frame_num_to_track=self.num_tracking_video_frames):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        vis_frame_stride = 1
        plt.close("all")
        out_obj_id = 1
        self.video_masks = [video_segments[i][out_obj_id] for i in range(self.num_tracking_video_frames)]
        mask = self.video_masks[self.sam_video_frame_idx]
        self.sam_mask.setScaleMask(self.sam_image_scale, mask.squeeze(0))
        self.sam_propagated_video = True
        self.update_sam()
        # for out_frame_idx in range(0, nun_frames, vis_frame_stride):
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(Image.open(os.path.join(self.jpg_dir, self.frame_names[out_frame_idx])))
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        # plt.show()

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        print("setEditing", value, self.mode)
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return
        self.prevMovePoint = pos
        self.restoreCursor()
        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode if "polygonSAM" != self.createMode else "polygon"

            self.overrideCursor(CURSOR_DRAW)
            if not self.current_prompts:
                if self.createMode == "polygonSAM":
                    points = np.array([[pos.x(), pos.y()]])
                    labels = np.array([1])
                    self.samPrompt(points, labels, click=False)
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current_prompts[-1], pos)
            elif (
                    self.snapping
                    and len(self.current_prompts) > 1
                    and self.createMode == "polygon"
                    and self.closeEnough(pos, self.current_prompts[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current_prompts[0]
                self.overrideCursor(CURSOR_POINT)
                self.current_prompts.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line[0] = self.current_prompts[-1]
                self.line[1] = pos
            elif self.createMode == "rectangle":
                self.line.points = [self.current_prompts[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current_prompts[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current_prompts[0], pos]
                self.line.close()
            elif self.createMode in "point":
                self.line.points = [self.current_prompts[0]]
                self.line.close()
            elif self.createMode == "polygonSAM":
                self.line.points = [pos, pos]
            self.repaint()
            self.current_prompts.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        print("left mouse event drawing, editing", self.drawing(), self.editing(), self.current_prompts,
              self.createMode)
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current_prompts:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current_prompts.addPoint(self.line[1])
                        self.line[0] = self.current_prompts[-1]
                        if self.current_prompts.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current_prompts.points) == 1
                        self.current_prompts.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current_prompts.addPoint(self.line[1])
                        self.line[0] = self.current_prompts[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == "polygonSAM":
                        self.current_prompts.addPoint(self.line[1], True)
                        points = [[point.x(), point.y()] for point in self.current_prompts.points]
                        labels = [int(label) for label in self.current_prompts.labels]
                        self.samPrompt(np.array(points), np.array(labels), click=True)
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    if self.createMode == "polygonSAM":
                        print("new polygonSAM")
                        self.current_prompts = MultipoinstShape()
                        self.current_prompts.addPoint(pos, True)
                    else:
                        self.current_prompts = Shape(shape_type=self.createMode)
                        self.current_prompts.addPoint(pos)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current_prompts.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                        self.selectedVertex()
                        and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton:
            if self.drawing() and self.createMode == "polygonSAM":
                if self.current_prompts:
                    self.current_prompts.addPoint(self.line[1], False)
                    points = [[point.x(), point.y()] for point in self.current_prompts.points]
                    labels = [int(label) for label in self.current_prompts.labels]
                    self.samPrompt(np.array(points), np.array(labels), click=True)
                    if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                        self.finalise()
                elif not self.outOfPixmap(pos):
                    self.current_prompts = MultipoinstShape()
                    self.current_prompts.addPoint(pos, False)
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()
            elif self.editing():
                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                if not self.selectedShapes or (
                        self.hShape is not None
                        and self.hShape not in self.selectedShapes
                ):
                    self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                    self.repaint()
                self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.createMode != "polygonSAM":
                menu = self.menus[len(self.selectedShapesCopy) > 0]
                self.restoreCursor()
                if (
                        not menu.exec_(self.mapToGlobal(ev.pos()))
                        and self.selectedShapesCopy
                ):
                    # Cancel the move by deleting the shadow copy.
                    self.selectedShapesCopy = []
                    self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                        self.hShape is not None
                        and self.hShapeIsSelected
                        and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        if self.createMode == "polygonSAM":
            print(
                f"self.drawing() and self.current and len(self.current) > 0 {self.drawing(), self.current_prompts, len(self.current_prompts) if self.current_prompts else 0}")
            return self.drawing() and (
                    self.video_masks is not None or (self.current_prompts and len(self.current_prompts) > 0))
        return self.drawing() and self.current_prompts and len(self.current_prompts) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
                self.double_click == "close"
                and self.canCloseShape()
                and len(self.current_prompts) > 3
        ):
            self.current_prompts.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self, selected=None):
        deleted_shapes = []
        if selected is None:
            to_delete = self.selectedShapes.copy()
            self.selectedShapes = []
        else:
            to_delete = selected
        if to_delete:
            for shape in to_delete:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        self.sam_mask.paint(p)

        # draw crosshair
        if (
                self._crosshair[self._createMode]
                and self.drawing()
                and self.prevMovePoint
                and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y()),
                self.width() - 1,
                int(self.prevMovePoint.y()),
            )
            p.drawLine(
                int(self.prevMovePoint.x()),
                0,
                int(self.prevMovePoint.x()),
                self.height() - 1,
            )

        Shape.scale = self.scale
        MultipoinstShape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(
                    shape
            ):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current_prompts:
            self.current_prompts.paint(p)
            self.line.paint(p)

        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
                self.fillDrawing()
                and self.createMode == "polygon"
                and self.current_prompts is not None
                and len(self.current_prompts.points) >= 2
        ):
            drawing_shape = self.current_prompts.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        print("finalize", self.createMode, self.current_prompts)

        if self.createMode == "polygonSAM":
            self.shapes.append(self.sam_mask)
        else:
            assert self.current_prompts
            self.current_prompts.close()
            self.shapes.append(self.current_prompts)
        self.storeShapes()
        self.sam_mask = MaskShape()
        self.current_prompts = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        elif ev.orientation() == QtCore.Qt.Vertical:
            mods = ev.modifiers()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                self.zoomRequest.emit(ev.delta(), ev.pos())
            else:
                self.scrollRequest.emit(
                    ev.delta(),
                    QtCore.Qt.Horizontal
                    if (QtCore.Qt.ShiftModifier == int(mods))
                    else QtCore.Qt.Vertical,
                )
        else:
            self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        key = ev.key()
        key_name = QtGui.QKeySequence(key).toString()
        try:
            print(f"canvas keyPressEvent {key_name} canvas self.drawing()", self.drawing())
        except:
            pass
        modifiers = ev.modifiers()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current_prompts:
                self.current_prompts = None
                self.drawingPolygon.emit(False)
                self.update()
            # enter key
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        # elif self.editing():
        # elif key == QtCore.Qt.Key_Left:
        #     self.open()
        #     self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
        # elif key == QtCore.Qt.Key_Right:
        #     self.openNextImg()
        # self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if (
                        self.shapesBackups[-1][index].points
                        != self.shapes[index].points
                ):
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False
        # isAutoRepeat=True only when actual releasing. Not when holding down.
        if self.hided_sam_mask is not None and not ev.isAutoRepeat():
            self.sam_unhide()

    def setLastLabel(self, text, flags):
        assert text
        polygons = None
        if len(self.shapes) > 0:
            self.shapes[-1].label = text
            self.shapes[-1].flags = flags
            if isinstance(self.shapes[-1], MaskShape):
                print("setLastLabel MaskShape")
                mask_shape = self.shapes.pop()
                polygons = mask_shape.toPolygons(self.sam_config["approxpoly_epsilon"])
                self.shapes.extend(polygons)
        self.shapesBackups.pop()
        self.storeShapes()
        if polygons is not None and isinstance(polygons, list):
            print("polygons", polygons)
            return polygons
        else:
            print("self.shapes", self.shapes)
            return self.shapes[-1:]

    def undoLastLine(self):
        assert self.shapes
        self.current_prompts = self.shapes.pop()
        self.current_prompts.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current_prompts[-1], self.current_prompts[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current_prompts.points = self.current_prompts.points[0:1]
        elif self.createMode in ["point", "polygonSAM"]:
            self.current_prompts = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current_prompts or self.current_prompts.isClosed():
            return
        self.current_prompts.popPoint()
        if len(self.current_prompts) > 0:
            self.line[0] = self.current_prompts[-1]
        else:
            self.current_prompts = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True, dir=None, file_name=None):
        self.img_dir = dir
        self.current_image_path = file_name
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
            self.sam_mask = MaskShape()
            self.current_prompts = None
        if self.createMode == "polygonSAM" and self.pixmap and self.sam_video_init:
            self.update_sam()
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current_prompts = None
        self.sam_mask = MaskShape()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()
