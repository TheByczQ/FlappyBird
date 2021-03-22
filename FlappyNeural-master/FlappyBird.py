import  pygame
import os
import random
import neat
import pickle
pygame.init()
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

pygame.display.set_caption('Flappy Network')
pygame.display.set_icon(pygame.image.load(os.path.join("images", "bird1.png")))

IMG_BIRD = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
IMG_PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
IMG_GROUND = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "ground.png")))
IMG_BACKGROUND = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "background.png")))
IMG_BACKGROUND_BLURED = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "background_blured.png")))
FONT_FLAPPY = pygame.font.Font(os.path.join("images", "FlappyBirdy.ttf"), 75)
FONT_SCORE = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMG = IMG_BIRD
    MAX_ROTATION = 25
    ROT_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMG[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.velocity*self.tick_count + 1.5*self.tick_count**2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION

        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VELOCITY

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMG[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMG[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMG[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMG[1]
        elif self.img_count < self.ANIMATION_TIME*4+1:
            self.img = self.IMG[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMG[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP_SIZE = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(IMG_PIPE, False, True)
        self.PIPE_BOTTOM = IMG_PIPE

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP_SIZE

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self,bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        B_collision_point = bird_mask.overlap(bottom_mask, bottom_offset)
        T_collision_point = bird_mask.overlap(top_mask, top_offset)

        if B_collision_point or T_collision_point:
            return True

        return False


class Ground:
    VELOCITY = 5
    WIDTH = IMG_GROUND.get_width()
    IMG = IMG_GROUND

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, ground, score, gen):
    win.blit(IMG_BACKGROUND, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = FONT_SCORE.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    text = FONT_SCORE.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    text = FONT_SCORE.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(text, (10, 50))
    for bird in birds:
        bird.draw(win)

    ground.draw(win)

    pygame.display.update()


def eval_genoms(genoms, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []
    j = False
    for _, g in genoms:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)

    ground = Ground(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    run = True
    clock = pygame.time.Clock()

    score = 0

    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    j = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    j = False
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5 or j:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        ground.move()
        draw_window(win, birds, pipes, ground, score, GEN)



def menu():
    IMG_BUTTON = pygame.image.load(os.path.join("images", "button.png"))

    print(pygame.font.get_fonts())
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    bird = Bird(100, 210)
    run = True
    while run:
        clock.tick(60)
        mousex, mousey = pygame.mouse.get_pos()
        button_switch = 0

        if 100 < mousex < 400:
            if 200 < mousey < 260:
                button_switch = 1
            elif 350 < mousey < 420:
                button_switch = 2
            elif 500 < mousey < 560:
                button_switch = 3
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_switch == 3:
                    run = False
        win.blit(IMG_BACKGROUND_BLURED, (0, 0))

        win.blit(IMG_BUTTON, (100, 200))
        win.blit(IMG_BUTTON, (100, 350))
        win.blit(IMG_BUTTON, (100, 500))

        if button_switch == 1:
            bird.y = 210
            bird.draw(win)
        elif button_switch == 2:
            bird.y = 360
            bird.draw(win)
        elif button_switch == 3:
            bird.y = 510
            bird.draw(win)

        text = FONT_FLAPPY.render("PLAY ", 1, (255, 255, 255))
        win.blit(text, (190 , 214))
        text2 = FONT_FLAPPY.render("LOAD NET ", 1, (255, 255, 255))
        win.blit(text2, (190, 364))
        text3 = FONT_FLAPPY.render("TRAIN NET ", 1, (255, 255, 255))
        win.blit(text3, (190, 514))

        pygame.display.update()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    cp = neat.checkpoint.Checkpointer()

    population = neat.Population(config)


    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    #pop = cp.restore_checkpoint("neat-checkpoint-5")
    #pop.run(eval_genoms)
    winner = population.run(eval_genoms)
    #cp.save_checkpoint(config, population, neat.DefaultSpeciesSet, GEN)

if __name__ == "__main__":
    menu()
    local_directory = os.path.dirname(__file__)
    config_path = os.path.join(local_directory, "network-config.txt")
    run(config_path)





