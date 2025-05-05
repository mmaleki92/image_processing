import cv2
import numpy as np
import mediapipe as mp
import math

class FingerPaint:
    def __init__(self):
        # Initialize mediapipe hand module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize drawing parameters
        self.drawing_color = (0, 0, 255)  # Start with red
        self.brush_thickness = 5
        self.eraser_thickness = 20
        self.is_eraser_mode = False
        
        # Will initialize canvas and toolbar after we know frame dimensions
        self.canvas = None
        self.toolbar = None
        self.frame_width = None
        self.frame_height = None
        self.toolbar_height = 100
        
        # Define colors for toolbar
        self.colors = [
            (0, 0, 255),   # Red
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 255, 255), # Yellow
            (255, 0, 255), # Purple
            (255, 255, 0), # Cyan
            (0, 0, 0),     # Black
            (255, 255, 255)# White/Eraser
        ]
        
        # Variables for tracking previous positions
        self.previous_x = 0
        self.previous_y = 0
        self.current_x = 0
        self.current_y = 0

    def create_toolbar(self):
        # Create color buttons in toolbar
        color_section_width = len(self.colors) * 50
        
        # Adjust start position based on toolbar width
        if color_section_width > self.frame_width:
            # If colors won't fit, scale down the button size
            button_width = self.frame_width // len(self.colors) - 10
            start_x = 5
        else:
            button_width = 40
            start_x = (self.frame_width - color_section_width) // 2
        
        # Clear toolbar before drawing
        self.toolbar.fill(0)
        
        for i, color in enumerate(self.colors):
            x_pos = start_x + i * (button_width + 10)
            cv2.rectangle(self.toolbar, (x_pos, 10), (x_pos + button_width, 90), color, -1)
            
            # Add border to active color button or eraser
            if (self.is_eraser_mode and i == len(self.colors) - 1) or \
               (not self.is_eraser_mode and color == self.drawing_color):
                cv2.rectangle(self.toolbar, (x_pos, 10), (x_pos + button_width, 90), (255, 255, 255), 2)

        # Add text label for eraser
        cv2.putText(self.toolbar, "ERASER", (start_x + (len(self.colors) - 1) * (button_width + 10) + 5, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        # Initialize canvas and toolbar if not yet created
        if self.canvas is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            self.toolbar = np.zeros((self.toolbar_height, self.frame_width, 3), dtype=np.uint8)
        
        # Flip frame horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and process drawing
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks for visualization
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                
                # Check if the finger is in the toolbar area
                if y < self.toolbar_height:
                    self.handle_toolbar_click(x)
                else:
                    # Get thumb tip position to detect drawing action
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    thumb_x = int(thumb_tip.x * frame.shape[1])
                    thumb_y = int(thumb_tip.y * frame.shape[0])
                    
                    # Calculate distance between index and thumb tips
                    distance = math.sqrt((x - thumb_x) ** 2 + (y - thumb_y) ** 2)
                    
                    # Drawing mode when index and thumb are close
                    if distance < 40:  # Adjust threshold as needed
                        # Calculate drawing position - use the actual finger position
                        if self.previous_x == 0 and self.previous_y == 0:
                            self.previous_x, self.previous_y = x, y
                        else:
                            canvas_y = y  # No longer subtracting toolbar_height
                            prev_canvas_y = self.previous_y  # No longer subtracting toolbar_height
                            
                            # Only draw if below toolbar area
                            if canvas_y > self.toolbar_height and prev_canvas_y > self.toolbar_height:
                                if self.is_eraser_mode:
                                    cv2.line(self.canvas, (self.previous_x, prev_canvas_y), 
                                             (x, canvas_y), (0, 0, 0), self.eraser_thickness)
                                else:
                                    cv2.line(self.canvas, (self.previous_x, prev_canvas_y), 
                                             (x, canvas_y), self.drawing_color, self.brush_thickness)
                            
                            self.previous_x, self.previous_y = x, y
                    else:
                        # Reset previous position if not drawing
                        self.previous_x, self.previous_y = 0, 0
        
        # Create merged frame with toolbar and canvas
        self.create_toolbar()
        merged_frame = frame.copy()
        
        # Apply canvas to frame using alpha blending
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # Create a blank background for the canvas part
        canvas_bg = np.zeros_like(frame)
        # Add canvas to the background
        canvas_bg_with_drawing = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        
        # Combine frame with canvas
        frame_without_drawing = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        merged_frame = cv2.add(frame_without_drawing, canvas_bg_with_drawing)
        
        # Add toolbar on top
        merged_frame[0:self.toolbar_height, 0:self.frame_width] = self.toolbar
        
        return merged_frame
    
    def handle_toolbar_click(self, x):
        # Calculate which color box was clicked
        color_section_width = len(self.colors) * 50
        
        # Adjust button size if needed
        if color_section_width > self.frame_width:
            button_width = self.frame_width // len(self.colors) - 10
            start_x = 5
        else:
            button_width = 40
            start_x = (self.frame_width - color_section_width) // 2
        
        for i, color in enumerate(self.colors):
            x_pos = start_x + i * (button_width + 10)
            if x_pos < x < x_pos + button_width:
                if i == len(self.colors) - 1:  # Last color is eraser
                    self.is_eraser_mode = True
                else:
                    self.is_eraser_mode = False
                    self.drawing_color = color
                break
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            cv2.imshow('Finger Paint', result)
            
            if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    app = FingerPaint()
    app.run()