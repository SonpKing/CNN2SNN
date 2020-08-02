from detection import SelectiveSearch, Camera, generate_boxes, generate_bb
from simulation.nest_sim import Simulator
import cv2 as cv

cam = Camera()
searcher = SelectiveSearch()
img = cam.get_frame()
rects = searcher.select(img)

sim = Simulator(scale=90, reset_sub=False)
sim.create_net("connections/", "input", "net.fc")
tics = 100
cls_pred = []
for rect in rects:
    inputs = generate_bb(img, rect)
    print(inputs.shape)
    sim.reset()
    sim.fit_input(inputs)
    sim.run(tics)
    cls_pred.append(sim.get_result())
print(cls_pred)
boxes, cls_inds, scores = generate_boxes(rects, cls_pred)
print(boxes)
print(cls_inds)
print(scores)
imOut = img.copy()
for rect in boxes:
    x, y, w, h = rect
    cv.rectangle( imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA )
cv.imshow( "Output", imOut )
