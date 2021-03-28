from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from _collections import deque
import matplotlib.pyplot as plt
import cv2


class Tracker:
   """
   This is the multi-target tracker.

   Parameters
   ----------
   metric : nn_matching.NearestNeighborDistanceMetric
       A distance metric for measurement-to-track association.
   max_age : int
       Maximum number of missed misses before a track is deleted.
   n_init : int
       Number of consecutive detections before the track is confirmed. The
       track state is set to `Deleted` if a miss occurs within the first
       `n_init` frames.

   Attributes
   ----------
   metric : nn_matching.NearestNeighborDistanceMetric
       The distance metric used for measurement to track association.
   max_age : int
       Maximum number of missed misses before a track is deleted.
   n_init : int
       Number of frames that a track remains in initialization phase.
   kf : kalman_filter.KalmanFilter
       A Kalman filter to filter target trajectories in image space.
   tracks : List[Track]
       The list of active tracks at the current time step.

   """

   def __init__(self, metric, max_iou_distance=0.7, max_age=60, n_init=3):
       self.metric = metric
       self.max_iou_distance = max_iou_distance
       self.max_age = max_age
       self.n_init = n_init

       self.kf = kalman_filter.KalmanFilter()
       self.tracks = []
       self._next_id = 1
       self.center = [deque(maxlen=30) for _ in range(1000)]
       self.cum_disp = [0 for _ in range(1000)]
       self.unique_count = 0
       self.current_count = 0
       self.counter = []

   def predict(self):
       """Propagate track state distributions one time step forward.

       This function should be called once every time step, before `update`.
       """
       for track in self.tracks:
           track.predict(self.kf)

   def update(self, detections):
       """Perform measurement update and track management.

       Parameters
       ----------
       detections : List[deep_sort.detection.Detection]
           A list of detections at the current time step.

       """
       # Run matching cascade.
       matches, unmatched_tracks, unmatched_detections = \
           self._match(detections)

       # Update track set.
       for track_idx, detection_idx in matches:
           self.tracks[track_idx].update(
               self.kf, detections[detection_idx])
       for track_idx in unmatched_tracks:
           self.tracks[track_idx].mark_missed()
       for detection_idx in unmatched_detections:
           self._initiate_track(detections[detection_idx])
       self.tracks = [t for t in self.tracks if not t.is_deleted()]

       # Update distance metric.
       active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
       features, targets = [], []
       for track in self.tracks:
           if not track.is_confirmed():
               continue
           features += track.features
           targets += [track.track_id for _ in track.features]
           track.features = []
       self.metric.partial_fit(
           np.asarray(features), np.asarray(targets), active_targets)

   def _match(self, detections):

       def gated_metric(tracks, dets, track_indices, detection_indices):
           features = np.array([dets[i].feature for i in detection_indices])
           targets = np.array([tracks[i].track_id for i in track_indices])
           cost_matrix = self.metric.distance(features, targets)
           cost_matrix = linear_assignment.gate_cost_matrix(
               self.kf, cost_matrix, tracks, dets, track_indices,
               detection_indices)

           return cost_matrix

       # Split track set into confirmed and unconfirmed tracks.
       confirmed_tracks = [
           i for i, t in enumerate(self.tracks) if t.is_confirmed()]
       unconfirmed_tracks = [
           i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

       # Associate confirmed tracks using appearance features.
       matches_a, unmatched_tracks_a, unmatched_detections = \
           linear_assignment.matching_cascade(
               gated_metric, self.metric.matching_threshold, self.max_age,
               self.tracks, detections, confirmed_tracks)

       # Associate remaining tracks together with unconfirmed tracks using IOU.
       iou_track_candidates = unconfirmed_tracks + [
           k for k in unmatched_tracks_a if
           self.tracks[k].time_since_update == 1]
       unmatched_tracks_a = [
           k for k in unmatched_tracks_a if
           self.tracks[k].time_since_update != 1]
       matches_b, unmatched_tracks_b, unmatched_detections = \
           linear_assignment.min_cost_matching(
               iou_matching.iou_cost, self.max_iou_distance, self.tracks,
               detections, iou_track_candidates, unmatched_detections)

       matches = matches_a + matches_b
       unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
       return matches, unmatched_tracks, unmatched_detections

   def _initiate_track(self, detection):
       mean, covariance = self.kf.initiate(detection.to_xyah())
       class_name = detection.get_class()
       self.tracks.append(Track(
           mean, covariance, self._next_id, self.n_init, self.max_age,
           detection.feature, class_name))
       self._next_id += 1

   def draw_box(self,frame):
       """draw boudning box for detected persons. Display counts of unique people
           and number of people moving left. Frame after adding text and box

       Parameters
       ----------
       self
       frame

       Returns
       ----------
       Frame
       """
       disp_tol = 3
       cmap = plt.get_cmap('tab20b')
       colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
       self.current_count = 0
       c_right=0 
       c_left=0
       for track in self.tracks:
           
           if not track.is_confirmed() or track.time_since_update > 1:
               continue 
           bbox = track.to_tlbr()
           class_name = track.get_class()
           
           # Track center movement
           center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
           self.center[track.track_id].append(center)
           
           color = colors[int(track.track_id) % len(colors)]
           color = [i * 255 for i in color]

           lr =''
           
           self.cum_disp[track.track_id] = 0
           disp = 0
           
           for j in range(1, len(self.center[track.track_id])):
               if self.center[track.track_id][j-1] is None or self.center[track.track_id][j] is None:
                   continue
               thickness = int(np.sqrt(64/float(j+1))*2)
               #cv2.line(frame, (self.center[track.track_id][j-1]), (self.center[track.track_id][j]), color, thickness)
          
           # Calculate center displacement
               disp = self.center[track.track_id][j][0]-self.center[track.track_id][j-1][0]
               self.cum_disp[track.track_id] += disp
           
           
           
           # draw bbox on screen
           if((bbox[2]-bbox[0])*(bbox[3]-bbox[1])<0.5*frame.shape[0]*frame.shape[1]):
               cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
               cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(5)*8, int(bbox[1])), color, -1)
               cv2.putText(frame, str(track.track_id)+lr,(int(bbox[0]), int(bbox[1]-10)),0, 0.5, (255,255,255),1)
           
           self.counter.append(int(track.track_id))
           self.current_count += 1
      
              
       # Count the number of tracks that have Total displacement > 0 or < 0
       t_lefts = 0
       t_rights = 0
       for k in range(len(self.cum_disp)):
           if self.cum_disp[k]<-disp_tol:
             t_lefts +=1
           elif self.cum_disp[k]>disp_tol:
             t_rights += 1
       print("Total Left Moves:",t_lefts)
       print("Total Right Moves:",t_rights)

       self.unique_count = len(set(self.counter))
       cv2.putText(frame, "Current Persons: " + str(self.current_count), (frame.shape[1]-200, 50), 0, 0.5, (0, 0, 255), 1)
       cv2.putText(frame, "Unique Persons: " + str(self.unique_count), (frame.shape[1]-200,80), 0, 0.5, (0,0,255), 1)
       cv2.putText(frame, "Total Left Moving: " + str(t_lefts), (frame.shape[1]-200, 110), 0, 0.5, (0, 0, 255), 1)
       
       return frame
