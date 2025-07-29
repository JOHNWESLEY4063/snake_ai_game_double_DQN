# ddqn_gui/snake_gameai.py
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
# Import GameConfig from the same ddqn_gui folder
from config import GameConfig, NetworkConfig

pygame.init()
try:
    font = pygame.font.Font('arial.ttf', 25)
except FileNotFoundError:
    font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 200, 0)

BLOCK_SIZE = GameConfig.BLOCK_SIZE
SPEED = GameConfig.SPEED

class SnakeGameAI:
    def __init__(self, enable_obstacles=False, w=GameConfig.WIDTH, h=GameConfig.HEIGHT):
        self.w = w
        self.h = h
        self.enable_obstacles = enable_obstacles

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.obstacles = []

        self._place_food()
        if self.enable_obstacles:
            self._place_static_obstacles()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def _place_static_obstacles(self):
        self.obstacles = []
        num_obstacles = GameConfig.NUM_STATIC_OBSTACLES
        for _ in range(num_obstacles):
            start_x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            start_y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            
            start_point = Point(start_x, start_y)
            if start_point in self.snake or start_point == self.food:
                continue

            direction_choice = random.choice([Direction.RIGHT, Direction.DOWN])
            length = random.randint(GameConfig.STATIC_OBSTACLE_LENGTH_MIN, GameConfig.STATIC_OBSTACLE_LENGTH_MAX)

            current_x, current_y = start_x, start_y
            for _ in range(length):
                obstacle_point = Point(current_x, current_y)
                if 0 <= obstacle_point.x < self.w and \
                   0 <= obstacle_point.y < self.h and \
                   obstacle_point not in self.snake and \
                   obstacle_point != self.food and \
                   obstacle_point not in self.obstacles:
                    self.obstacles.append(obstacle_point)
                
                if direction_choice == Direction.RIGHT:
                    current_x += BLOCK_SIZE
                elif direction_choice == Direction.DOWN:
                    current_y += BLOCK_SIZE
                
                if not (0 <= current_x < self.w and 0 <= current_y < self.h):
                    break
        
        if self.food in self.obstacles:
            self._place_food()

    def get_state(self):
        head = self.head
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_ul = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_dl = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        point_dr = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        danger_straight = False
        danger_right_turn = False
        danger_left_turn = False
        danger_back_turn = False

        if dir_r:
            danger_straight = self.is_collision(point_r)
            danger_right_turn = self.is_collision(point_d)
            danger_left_turn = self.is_collision(point_u)
            danger_back_turn = self.is_collision(point_l)
        elif dir_l:
            danger_straight = self.is_collision(point_l)
            danger_right_turn = self.is_collision(point_u)
            danger_left_turn = self.is_collision(point_d)
            danger_back_turn = self.is_collision(point_r)
        elif dir_u:
            danger_straight = self.is_collision(point_u)
            danger_right_turn = self.is_collision(point_r)
            danger_left_turn = self.is_collision(point_l)
            danger_back_turn = self.is_collision(point_d)
        elif dir_d:
            danger_straight = self.is_collision(point_d)
            danger_right_turn = self.is_collision(point_l)
            danger_left_turn = self.is_collision(point_r)
            danger_back_turn = self.is_collision(point_u)

        if dir_r:
            danger_fr_diag = self.is_collision(point_ur)
            danger_br_diag = self.is_collision(point_dr)
            danger_fl_diag = self.is_collision(point_ul)
            danger_bl_diag = self.is_collision(point_dl)
        elif dir_l:
            danger_fr_diag = self.is_collision(point_dl)
            danger_br_diag = self.is_collision(point_ul)
            danger_fl_diag = self.is_collision(point_dr)
            danger_bl_diag = self.is_collision(point_ur)
        elif dir_u:
            danger_fr_diag = self.is_collision(point_ur)
            danger_br_diag = self.is_collision(point_ul)
            danger_fl_diag = self.is_collision(point_dr)
            danger_bl_diag = self.is_collision(point_dl)
        elif dir_d:
            danger_fr_diag = self.is_collision(point_dl)
            danger_br_diag = self.is_collision(point_dr)
            danger_fl_diag = self.is_collision(point_ul)
            danger_bl_diag = self.is_collision(point_ur)
        else:
            danger_fr_diag, danger_br_diag, danger_fl_diag, danger_bl_diag = False, False, False, False

        state = [
            danger_straight,
            danger_right_turn,
            danger_left_turn,
            danger_back_turn,
            danger_fr_diag,
            danger_fl_diag,
            danger_br_diag,
            danger_bl_diag,

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y
        ]
        return np.array(state, dtype=int)

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or \
           self.frame_iteration > GameConfig.COLLISION_TIMEOUT_MULTIPLIER * len(self.snake):
            game_over = True
            reward = GameConfig.REWARD_COLLISION
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = GameConfig.REWARD_FOOD
            self._place_food()
        else:
            old_dist = math.hypot(self.snake[1].x - self.food.x, self.snake[1].y - self.food.y)
            new_dist = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)

            if new_dist < old_dist:
                reward = GameConfig.REWARD_CLOSER
            elif new_dist > old_dist:
                reward = GameConfig.REWARD_FURTHER
            else:
                reward = GameConfig.REWARD_STEP

            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if self.enable_obstacles and pt in self.obstacles:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=3)
            pygame.draw.rect(self.display, (50, 50, 255), pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), border_radius=2)

        head_x, head_y = self.head.x, self.head.y
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(head_x, head_y, BLOCK_SIZE, BLOCK_SIZE), border_radius=5)
        pygame.draw.rect(self.display, (100, 150, 255), pygame.Rect(head_x + 2, head_y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), border_radius=4)

        eye_radius = BLOCK_SIZE // 8
        eye_offset_x = BLOCK_SIZE // 4
        eye_offset_y = BLOCK_SIZE // 4

        if self.direction == Direction.RIGHT:
            left_eye_pos = (head_x + BLOCK_SIZE - eye_offset_x, head_y + eye_offset_y)
            right_eye_pos = (head_x + BLOCK_SIZE - eye_offset_x, head_y + BLOCK_SIZE - eye_offset_y)
        elif self.direction == Direction.LEFT:
            left_eye_pos = (head_x + eye_offset_x, head_y + eye_offset_y)
            right_eye_pos = (head_x + eye_offset_x, head_y + BLOCK_SIZE - eye_offset_y)
        elif self.direction == Direction.UP:
            left_eye_pos = (head_x + eye_offset_y, head_y + eye_offset_x)
            right_eye_pos = (head_x + BLOCK_SIZE - eye_offset_y, head_y + eye_offset_x)
        elif self.direction == Direction.DOWN:
            left_eye_pos = (head_x + eye_offset_y, head_y + BLOCK_SIZE - eye_offset_x)
            right_eye_pos = (head_x + BLOCK_SIZE - eye_offset_y, head_y + BLOCK_SIZE - eye_offset_x)
        else:
            left_eye_pos = (head_x + eye_offset_x, head_y + eye_offset_y)
            right_eye_pos = (head_x + BLOCK_SIZE - eye_offset_x, head_y + eye_offset_y)

        pygame.draw.circle(self.display, WHITE, left_eye_pos, eye_radius)
        pygame.draw.circle(self.display, WHITE, right_eye_pos, eye_radius)
        pygame.draw.circle(self.display, BLACK, left_eye_pos, eye_radius // 2)
        pygame.draw.circle(self.display, BLACK, right_eye_pos, eye_radius // 2)

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=5)
        pygame.draw.circle(self.display, (255, 100, 100), (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2), BLOCK_SIZE // 3)

        if self.enable_obstacles:
            for pt in self.obstacles:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=3)
                pygame.draw.rect(self.display, (0, 150, 0), pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), border_radius=2)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

if __name__ == '__main__':
    game = SnakeGameAI(enable_obstacles=True)
    while True:
        action_for_test = [1, 0, 0]
        reward, done, score = game.play_step(action_for_test)
        if done:
            print(f'Final Score: {score}')
            game.reset()
