import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class PressMachineSimulator:
    def __init__(self, area_threshold=15000):
        # Simulation parameters
        self.frame_size = (400, 600)  # height, width
        self.rect_width = 200
        self.min_rect_height = 50
        self.max_rect_height = 200
        self.area_threshold = area_threshold
        
        # Tracking variables
        self.press_count = 0
        self.prev_press_state = "OPEN"
        self.areas = []
        self.states = []
        self.frame_count = 0
        
        # Setup visualization
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Press Machine Simulation with Area Analysis", fontsize=16)
        
        # Image display
        self.axs[0, 0].set_title("Current Frame")
        self.img_display = self.axs[0, 0].imshow(np.zeros((self.frame_size[0], self.frame_size[1], 3)))
        self.axs[0, 0].axis('off')
        
        # Contour display
        self.axs[0, 1].set_title("Contour Detection")
        self.contour_display = self.axs[0, 1].imshow(np.zeros((self.frame_size[0], self.frame_size[1], 3)))
        self.axs[0, 1].axis('off')
        
        # Area plot
        self.axs[1, 0].set_title("Contour Area Over Time")
        self.axs[1, 0].set_xlabel("Frame")
        self.axs[1, 0].set_ylabel("Area")
        self.axs[1, 0].grid(True)
        self.area_line, = self.axs[1, 0].plot([], [], lw=2)
        self.threshold_line = self.axs[1, 0].axhline(y=self.area_threshold, color='r', linestyle='--', label="Threshold")
        self.axs[1, 0].legend()
        
        # Press state plot
        self.axs[1, 1].set_title("Press State Over Time")
        self.axs[1, 1].set_xlabel("Frame")
        self.axs[1, 1].set_yticks([0, 1])
        self.axs[1, 1].set_yticklabels(['OPEN', 'CLOSED'])
        self.axs[1, 1].grid(True)
        self.state_line, = self.axs[1, 1].plot([], [], drawstyle='steps-post', lw=2)
        
        # Text annotations - Use ax.text instead of fig.text for better blitting
        self.press_text = self.axs[0, 0].text(10, 10, "Press Count: 0", 
                                      fontsize=12, color='red', 
                                      bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

    def create_press_frame(self, rect_height, rect_y_position):
        """Create a single frame with a rectangle"""
        frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 255
        rect_x = (self.frame_size[1] - self.rect_width) // 2
        cv2.rectangle(frame, (rect_x, rect_y_position), 
                     (rect_x + self.rect_width, rect_y_position + rect_height), 
                     (0, 0, 0), -1)
        return frame
    
    def detect_press(self, frame):
        """Detect and analyze the press in the frame"""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours if found
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Create visualization frame
            vis_frame = frame.copy()
            cv2.drawContours(vis_frame, [largest_contour], -1, (0, 0, 255), 2)
            
            # Add area information
            cv2.putText(vis_frame, f"Area: {int(area)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Determine press state
            if area > self.area_threshold:
                press_state = "CLOSED"
                state_color = (0, 0, 255)  # Red for closed
            else:
                press_state = "OPEN"
                state_color = (0, 255, 0)  # Green for open
            
            # Add press state
            cv2.putText(vis_frame, f"Press: {press_state}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            return frame, vis_frame, area, press_state
        
        return frame, frame.copy(), 0, "OPEN"
    
    def update(self, frame_num):
        """Update function for animation"""
        self.frame_count += 1
        
        # Calculate rectangle parameters based on sine wave to simulate press
        phase = (self.frame_count % 60) / 60  # 60-frame cycle
        if phase < 0.5:
            # Press going down
            completion = phase * 2  # 0 to 1 in first half
            rect_height = int(self.min_rect_height + completion * (self.max_rect_height - self.min_rect_height))
        else:
            # Press going up
            completion = (phase - 0.5) * 2  # 0 to 1 in second half
            rect_height = int(self.max_rect_height - completion * (self.max_rect_height - self.min_rect_height))
        
        rect_y = 100  # Fixed top position
        
        # Create and analyze frame
        frame = self.create_press_frame(rect_height, rect_y)
        orig_frame, vis_frame, area, press_state = self.detect_press(frame)
        
        # Update area and state history
        self.areas.append(area)
        state_num = 1 if press_state == "CLOSED" else 0
        self.states.append(state_num)
        
        # Count presses
        if self.prev_press_state == "OPEN" and press_state == "CLOSED":
            self.press_count += 1
            self.press_text.set_text(f"Press Count: {self.press_count}")
        
        self.prev_press_state = press_state
        
        # Update visualization
        self.img_display.set_array(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
        self.contour_display.set_array(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        frame_indices = list(range(len(self.areas)))
        self.area_line.set_data(frame_indices, self.areas)
        self.state_line.set_data(frame_indices, self.states)
        
        # Auto-adjust area plot limits
        if self.areas:
            max_area = max(max(self.areas) * 1.1, self.area_threshold * 1.5)
            self.axs[1, 0].set_xlim(0, max(1, len(self.areas)))
            self.axs[1, 0].set_ylim(0, max_area)
        
        # Auto-adjust state plot limits
        self.axs[1, 1].set_xlim(0, max(1, len(self.states)))
        self.axs[1, 1].set_ylim(-0.1, 1.1)
        
        # Only return the artists that we're updating
        return [self.img_display, self.contour_display, self.area_line, self.state_line, self.press_text]

    def run_animation(self, frames=300):
        """Run the animation"""
        ani = FuncAnimation(self.fig, self.update, frames=frames, 
                           interval=50, blit=True)
        plt.show()
        return ani

# Run the simulation
if __name__ == "__main__":
    simulator = PressMachineSimulator(area_threshold=15000)
    ani = simulator.run_animation(frames=300)