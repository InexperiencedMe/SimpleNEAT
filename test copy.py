import stable_retro as retro
import gymnasium as gym
import numpy as np
import cv2
import pygame

class CleanMario(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.actionSize = 6 
        self.observationShape = (8, 8)
        self.observationSize = np.prod(self.observationShape)
        
        originalShape = env.observation_space.shape
        self.cropTop = int(originalShape[0] * 0.5)
        self.cropLeft = int(originalShape[1] * 0.5)
        
        self.maxXposition = 0
        self.stagnationCounter = 0

    def getRAMvalues(self):
        ram = self.env.unwrapped.get_ram()
        xPosition = int(ram[0x94]) + (int(ram[0x95]) * 256) 
        
        isDead = (ram[0x71] == 9)   
        
        # --- THE FIX ---
        # 0x1493: End Level Timer (becomes > 0 when hitting tape or boss sphere)
        # 0x13CE: Level Beaten Flag (128 / 0x80 is the actual beaten flag. 64 / 0x40 is the Midway gate)
        # 0x71 == 12: Player Animation State for Castle Walk / Goal Tape
        isBeaten = (ram[0x1493] > 0) or (ram[0x13CE] >= 128) or (ram[0x71] == 12)                          
        return xPosition, isDead, isBeaten

    def processObservation(self, obs):
        croppedObs = obs[self.cropTop:, self.cropLeft:]
        greenObs = croppedObs[:, :, 1] 
        targetSize = (self.observationShape[1], self.observationShape[0])
        resizedObs = cv2.resize(greenObs, targetSize, interpolation=cv2.INTER_AREA)
        
        return (resizedObs / 127.5) - 1.0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        xPosition, _, _ = self.getRAMvalues()
        self.maxXposition = xPosition
        self.stagnationCounter = 0
        
        return self.processObservation(obs)

    def step(self, action):
        fullAction = [0] * 12
        fullAction[0] = 1 if action[0] > 0 else 0 # B (Jump)
        fullAction[1] = 1 if action[1] > 0 else 0 # Y (Run/Shoot)
        fullAction[5] = 1 if action[2] > 0 else 0 # Down
        fullAction[6] = 1 if action[3] > 0 else 0 # Left
        fullAction[7] = 1 if action[4] > 0 else 0 # Right
        fullAction[8] = 1 if action[5] > 0 else 0 # A (Spin Jump)

        obs, _, terminated, truncated, info = self.env.step(fullAction)
        reward = 0

        currentX, isDead, isBeaten = self.getRAMvalues()
        progress = currentX - self.maxXposition
        
        if progress > 0:
            self.maxXposition = currentX
            self.stagnationCounter = 0
            reward += float(progress)
        else:
            self.stagnationCounter += 1
            
        reward -= 0.25 # Reduced constant penalty from 1 to 0.25

        if isBeaten:
            reward += 1000
            
        # Increased stagnation limit to 5 seconds (60 frames/sec * 5 = 300)
        done = terminated or truncated or isDead or isBeaten or self.stagnationCounter >= 300

        return self.processObservation(obs), reward, done
    
def environmentMaker(render_mode="rgb_array"): # <-- Changed to rgb_array
    return CleanMario(retro.make("SuperMarioWorld-Snes-v0", state="DonutPlains1", render_mode=render_mode))

def play_manually():
    pygame.init()
    
    env = environmentMaker(render_mode="rgb_array")
    env.reset()
    
    # SNES internal resolution is 256x224. We scale it x2 for visibility.
    scale = 2
    screen_width, screen_height = 256 * scale, 224 * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mario on WSL (Custom Pygame Renderer)")
    
    clock = pygame.time.Clock()
    total_reward = 0
    done = False
    
    print("="*50)
    print("Controls:")
    print("Arrow Keys : Move / Duck")
    print("Z          : Jump (B button)")
    print("X          : Run (Y button)")
    print("C          : Spin Jump (A button)")
    print("ESC        : Quit")
    print("="*50)

    while not done:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_ESCAPE]:
            print("Quitting early...")
            break

        # Map keyboard
        action = [0, 0, 0, 0, 0, 0]
        if keys[pygame.K_z]: action[0] = 1
        if keys[pygame.K_x]: action[1] = 1
        if keys[pygame.K_DOWN]: action[2] = 1
        if keys[pygame.K_LEFT]: action[3] = 1
        if keys[pygame.K_RIGHT]: action[4] = 1
        if keys[pygame.K_c]: action[5] = 1

        # Step environment
        obs, reward, done = env.step(action)
        total_reward += reward
        
        # --- RENDER TO PYGAME DIRECTLY ---
        # env.render() returns a numpy array of shape (height, width, 3)
        frame = env.render()
        
        # Pygame expects (width, height, 3), so we transpose the first two axes
        frame = np.transpose(frame, (1, 0, 2))
        
        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(frame)
        
        # Scale and draw onto screen
        surf = pygame.transform.scale(surf, (screen_width, screen_height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward > 0:
            print(f"Step Reward: {reward:7.2f} | Total Reward: {total_reward:7.2f}")
            
        clock.tick(60)
        
    print(f"\nGame Over! Final Total Reward: {total_reward:.2f}")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_manually()