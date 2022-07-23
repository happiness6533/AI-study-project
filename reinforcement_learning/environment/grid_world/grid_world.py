import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.unit = 100  # 픽셀 수
        self.height = 5  # 그리드 월드 가로
        self.width = 5  # 그리드 월드 세로
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('그리드 월드')
        self.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    @staticmethod
    def load_images():
        rectangle = PhotoImage(
            Image.open(
                "C:/Users/happiness/developer/machine_learning_study/reinforcement_learning/environment/grid_world/img/rectangle.png").resize(
                (65, 65)))
        triangle = PhotoImage(
            Image.open(
                "C:/Users/happiness/developer/machine_learning_study/reinforcement_learning/environment/grid_world/img/triangle.png").resize(
                (65, 65)))
        circle = PhotoImage(
            Image.open(
                "C:/Users/happiness/developer/machine_learning_study/reinforcement_learning/environment/grid_world/img/circle.png").resize(
                (65, 65)))

        return rectangle, triangle, circle

    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=self.height * self.unit,
                           width=self.width * self.unit)
        # 그리드 생성
        for c in range(0, self.width * self.unit, self.unit):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, self.height * self.unit
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * self.unit, self.unit):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, self.height * self.unit, r
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, self.unit / 2 - x, self.unit / 2 - y)
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # 상
            if state[1] > self.unit:
                base_action[1] -= self.unit
        elif action == 1:  # 하
            if state[1] < (self.height - 1) * self.unit:
                base_action[1] += self.unit
        elif action == 2:  # 좌
            if state[0] > self.unit:
                base_action[0] -= self.unit
        elif action == 3:  # 우
            if state[0] < (self.width - 1) * self.unit:
                base_action[0] += self.unit
        # 에이전트 이동
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # 에이전트(빨간 네모)를 가장 상위로 배치
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)

        # 보상 함수
        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2)]:
            reward = -100
            done = True
        else:
            reward = -1
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (self.unit * col), origin_x + (self.unit * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for x in range(self.height):
            for y in range(self.width):
                for action in range(0, 4):
                    state = [x, y]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(y, x, round(temp, 2), action)
