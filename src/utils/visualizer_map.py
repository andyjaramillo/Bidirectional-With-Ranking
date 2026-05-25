import pygame
from pygame_files.drawGame import draw_game
from src.environment.SokobanGame import SokobanGame
pygame.init()
screen = pygame.display.set_mode((640, 640))
clock = pygame.time.Clock()
running = True
string_to_visualize = "0000000000011111111001143111100014111110012111111001101004100000000000000000000000000000000000000000"
p = SokobanGame.decodeMap(string_to_visualize)
draw_game(p, screen)
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()