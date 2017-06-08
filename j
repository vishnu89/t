# from common_utils import *
from projectConfig import *
np.set_printoptions(2,suppress=True)
from face_recogontion.faceDetector import get_face_points, get_dlib_face_rect
scale_factor = 1
face_tracker_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
# from matplotlib import pyplot as plt
class Spider:
    def __init__(self,frameno, img, rect):
        x, y, w, h = p2r(rect)
        self.winSize = int(max(w,h))
        self.frameno = frameno
        self.gframe0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.p0 = np.array(get_face_points(img, p2d(rect), face_tracker_points)[0]).astype('f4')
        self.centroid0 = self.p0.mean(axis=0)
        # self.fig, self.ax = plt.subplots()
        self.il = ImgLog('constant',size=(512,512),col=1)
        self.p0_bk = []


    def display(self, frame):
        frame = put_text(frame, self.frameno)
        amp = self.amp
        motion1,colors = self.motion_analytics(amp)
        # self.il.log(frame)
        for pt,color in zip(self.p1,colors):
            # cv2.arrowedLine(frame, tuple(self.centroid1), tuple(pt), color=color)
            cv2.circle(frame,tuple(pt),3,(255,255,255),-1)
        # amp = np.abs(self.amp - self.amp.mean())
        # motion2 = self.motion_analytics(amp)

        self.il.log(motion1)
        # self.il.log(motion2)
        img = self.il.get()
        imshow("self.il.get() display 31 spider_tracker_nitish2", img)
        return frame

    def motion_analytics(self,amp):
        colors = np.tile((255,0,0),(len(amp),1))
        colors[amp>self.winSize] = (0,0,255)
        motion1 = cvbar(amp, scale=10, height=1000, size=(512, 512),colors=colors)
        motion1 = put_text(motion1, amp.std())
        return motion1,colors


    def pre(self):
        # self.winSize = 15
        self.lk_params = dict(winSize=(self.winSize, self.winSize), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        pass

    def update(self,frameno, img):
        self.frameno = frameno
        self.gframe1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.pre()
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.gframe0, self.gframe1, self.p0, None, **self.lk_params)
        self.centroid1 = self.p1.mean(axis=0)
        self.stabilizer()

    def stabilizer(self):
        self.amp = np.array([get_dist(o_pt, n_pt) for o_pt, n_pt in zip(self.p0, self.p1)])
        self.phs = np.array([get_angle(o_pt, n_pt) for o_pt, n_pt in zip(self.p0,self.p1)])
        self.magnitude_voting(self.amp)

    def angle_voting(self, phs,c):
        '''
        val = [1, 2, 3, 4, 360, 357, 344, 359, 359, 359, 359, 358]*100
        print self.angle_voting(np.array(val))
        :param flowVec:
        :return:
        '''
        flowVec = phs.copy()
        flowVec[~c] = 999
        imaxvote = (0 <= flowVec) & (flowVec < 90) | (flowVec == 360)
        q2 = (90 <= flowVec) & (flowVec < 180)
        q3 = (180 <= flowVec) & (flowVec < 270)
        q4 = (270 <= flowVec) & (flowVec < 360)
        maxvote = np.count_nonzero(imaxvote)
        cq2 = np.count_nonzero(q2)
        cq3 = np.count_nonzero(q3)
        cq4 = np.count_nonzero(q4)
        if cq2 > maxvote: imaxvote = q2; maxvote = cq2
        if cq3 > maxvote: imaxvote = q3; maxvote = cq3
        if cq4 > maxvote: imaxvote = q4; maxvote = cq4
        return imaxvote

    def magnitude_voting(self, motion):
        '''
        motion = range(20)
        motion = np.array(motion)
        print magnitue_voting(motion)
        '''
        # Val = motion.mean()+1*motion.std()
        # Val = motion.mean()/.7
        zero_th = 4
        slow_th = self.winSize
        fast_th = self.winSize
        zero_motion = motion <= zero_th  # discard zero motion vectors
        slow_motion = (zero_th < motion) & (motion <= slow_th)
        fast_motion = (slow_th < motion) & (motion <= fast_th)
        # print np.round(motion[zero_motion],2).tolist()
        # print np.round(motion[slow_motion],2).tolist()
        # print np.round(motion[fast_motion],2).tolist()
        return
        invalid = fast_th < motion
        if zero_motion.all():
            return 'zero', np.array([]), np.array([]) , np.array([]) # 'zero_motion',
        cslow, cfast = np.count_nonzero(slow_motion), np.count_nonzero(fast_motion)
        cvalid = float(cslow + cfast)
        if not cvalid:
            motiontype = 'slow'
            return motiontype, slow_motion, fast_motion, invalid
        if cslow/cvalid >= .5:
            motiontype = 'slow'
        if cfast/cvalid >= .5:
            motiontype = 'fast'
        self.th = cslow/cvalid, cfast/cvalid
        return motiontype, slow_motion, fast_motion, invalid

    def post(self):
        self.p0_bk.append(self.p0)
        self.p0 = self.p1
        self.gframe0 = self.gframe1
        self.centroid0 = self.centroid1
        (a,b),(c,d) = self.p0.min(axis=0), self.p0.max(axis=0)
        x,y,w,h = p2r((a,b,c,d))
        self.winSize = int(max(w,h))


if __name__ == '__main__':
    def labeller(img):
        return img.replace('i_','').replace('.jpg','').split('_')[1]


    setup = True
    video = RunVideo(r'/home/dondana/vishnu/workspace_py/video/face_crossing/*',name2label=labeller,queue_size=5, start_frame=0, resize=1, wait_time=1)
    # video = RunVideo(r'C:\Users\donDana\Downloads\VadiveluBankComedy.mp4', start_frame=115, resize=.25, wait_time=1)
    # video = RunVideo(r'/home/u826828/vishnu/dataBase/videos/reEntry.avi', start_frame=49, resize=1, wait_time=0)
    name = dict(ruppin=2,sameer=0,apurba=1,nitish=3)
    for chunks in video.img_as_queue():
        fno,imgs = zip(*chunks)
        imgs = np.array(imgs)
        img1 = np.rollaxis(imgs, 1)
        img2 = np.flip(img1,1)
        img = np.column_stack([img1,img2])
        img = np.vstack(img)
        # if setup:
        #     setup = False
        #     # rect = get_dlib_face_rect(img)[name['apurba']]
        #     rect = get_dlib_face_rect(imgs[0])[name['sameer']]
        #     # rect = get_dlib_face_rect(img)[name['ruppin']]
        #     # rect = get_dlib_face_rect(img)[name['nitish']]
        #     t = Spider(fno, img,np.array(rect)*5)
        #     tic = clk()
        # t.update(fno, img)
        # img = t.display(img)
        imshow("img",img.copy(),1)
        # t.post()
    # print fno/float(tic.toc())
    # print t.p0_bk
