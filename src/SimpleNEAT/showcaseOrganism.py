import pygame as pg

def showcaseOrganism(organism, environmentMaker, episodes=10, fps=60, upscalingFactor=4):
    pg.init()
    screen = None
    clock = pg.time.Clock()
    environment = environmentMaker(render_mode="rgb_array")
    for i in range(episodes):
        state = environment.reset()

        frame = environment.render()
        frameHeight, frameWidth, _ = frame.shape
        targetHeight, targetWidth = frameHeight*upscalingFactor, frameWidth*upscalingFactor

        organism.clearMemory()
        done = False
        score = 0
        while not done:
            if screen is None: screen = pg.display.set_mode((targetWidth, targetHeight))

            frame = environment.render()
            surface = pg.surfarray.make_surface(frame.swapaxes(0, 1))
            upscaledSurface = pg.transform.scale(surface, (targetWidth, targetHeight))
            screen.blit(upscaledSurface, (0, 0))
            pg.display.flip()

            action = organism(state)
            state, reward, done = environment.step(action)
            score += reward
            
            for event in pg.event.get():
                if event.type == pg.QUIT: return

            clock.tick(fps)
        print(f"Showcase Episode {i+1}. Score: {score:>8.2f}")
    pg.quit()
