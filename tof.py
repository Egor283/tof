#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
import math
import numpy as np

class TofSimulator(Node):
    def __init__(self):
        super().__init__('tof_simulator')
        
        # Параметры симуляции
        self.declare_parameter('update_rate', 50.0)  # Гц
        self.declare_parameter('min_range', 0.03)    # метры
        self.declare_parameter('max_range', 2.0)     # метры
        self.declare_parameter('fov', 0.26)          # радианы (примерно 15 градусов)
        self.declare_parameter('sensor_offset', 0.1)  # расстояние от центра до датчика (м)
        
        # Состояние робота (виртуальное)
        self.robot_x = 0.0      # позиция по X
        self.robot_y = 0.0      # позиция по Y
        self.robot_theta = 0.0  # ориентация (рад)
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Препятствия (виртуальные стены для демонстрации)
        self.obstacles = [
            {'x': 1.5, 'y': 0.0, 'radius': 0.3},  # круглое препятствие
            {'x': 3.0, 'y': 1.0, 'radius': 0.4},
            {'x': 2.0, 'y': -1.5, 'radius': 0.35},
        ]
        
        # Таймер для обновления позиции (симуляция движения)
        self.create_timer(0.01, self.update_robot_pose)  # 100 Hz для плавности
        
        # Подписка на команды скорости (если нужно реальное управление)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Создаем издателей для датчиков
        update_period = 1.0 / self.get_parameter('update_rate').value
        self.left_pub = self.create_publisher(Range, '/tof/left', 10)
        self.right_pub = self.create_publisher(Range, '/tof/right', 10)
        
        # Таймер для публикации данных датчиков
        self.create_timer(update_period, self.publish_sensor_data)
        
        self.get_logger().info('ToF симулятор запущен')
        
    def cmd_vel_callback(self, msg):
        """Обновление скорости из топика cmd_vel"""
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z
        
    def update_robot_pose(self):
        """Обновление позиции робота на основе скорости"""
        dt = 0.01  # соответствует периоду таймера
        
        # Обновляем позицию
        self.robot_x += self.linear_vel * math.cos(self.robot_theta) * dt
        self.robot_y += self.linear_vel * math.sin(self.robot_theta) * dt
        self.robot_theta += self.angular_vel * dt
        
        # Нормализация угла
        self.robot_theta = math.atan2(math.sin(self.robot_theta), 
                                      math.cos(self.robot_theta))
        
    def calculate_distance_to_obstacle(self, sensor_angle_offset):
        """
        Рассчитывает расстояние до ближайшего препятствия для датчика
        sensor_angle_offset: смещение угла датчика относительно направления робота
        левый датчик: +offset, правый: -offset
        """
        sensor_offset = self.get_parameter('sensor_offset').value
        max_range = self.get_parameter('max_range').value
        min_range = self.get_parameter('min_range').value
        
        # Позиция датчика (смещение от центра робота)
        sensor_x = self.robot_x + sensor_offset * math.cos(self.robot_theta + sensor_angle_offset)
        sensor_y = self.robot_y + sensor_offset * math.sin(self.robot_theta + sensor_angle_offset)
        
        # Направление луча датчика (немного смещено наружу для реализма)
        ray_angle = self.robot_theta + sensor_angle_offset
        
        # Поиск ближайшего препятствия
        min_dist = max_range
        
        for obs in self.obstacles:
            # Вектор от датчика до препятствия
            dx = obs['x'] - sensor_x
            dy = obs['y'] - sensor_y
            
            # Расстояние до центра препятствия
            dist_to_center = math.sqrt(dx*dx + dy*dy)
            
            # Проекция на направление луча
            ray_dot = dx * math.cos(ray_angle) + dy * math.sin(ray_angle)
            
            if ray_dot > 0:  # Препятствие впереди по лучу
                # Расстояние до пересечения с окружностью
                # (упрощенно - используем расстояние до центра минус радиус)
                dist = max(0.0, dist_to_center - obs['radius'])
                
                # Проверяем, попадает ли препятствие в поле зрения
                angle_to_obs = math.atan2(dy, dx)
                angle_diff = abs(angle_to_obs - ray_angle)
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                
                if angle_diff < self.get_parameter('fov').value:
                    if dist < min_dist:
                        min_dist = dist
        
        # Добавляем граничные стены для более предсказуемого поведения
        wall_dist = self.check_walls(sensor_x, sensor_y, ray_angle)
        if wall_dist < min_dist:
            min_dist = wall_dist
        
        # Ограничиваем диапазоном измерений
        if min_dist < min_range:
            min_dist = min_range
        elif min_dist > max_range:
            min_dist = float('inf')  # за пределами диапазона
            
        return min_dist
    
    def check_walls(self, x, y, angle):
        """Проверка расстояния до виртуальных стен"""
        max_range = self.get_parameter('max_range').value
        
        # Параметры комнаты
        room_size = 5.0
        walls = [
            {'type': 'x', 'pos': room_size, 'normal': -1},  # правая стена
            {'type': 'x', 'pos': -room_size, 'normal': 1},  # левая стена
            {'type': 'y', 'pos': room_size, 'normal': -1},  # передняя стена
            {'type': 'y', 'pos': -room_size, 'normal': 1},  # задняя стена
        ]
        
        min_dist = float('inf')
        
        for wall in walls:
            if wall['type'] == 'x':
                # Вертикальная стена
                if abs(math.cos(angle)) > 1e-6:  # избегаем деления на ноль
                    t = (wall['pos'] - x) / math.cos(angle)
                    if t > 0:
                        y_hit = y + t * math.sin(angle)
                        if -room_size <= y_hit <= room_size:
                            if t < min_dist:
                                min_dist = t
            else:  # 'y' - горизонтальная стена
                if abs(math.sin(angle)) > 1e-6:
                    t = (wall['pos'] - y) / math.sin(angle)
                    if t > 0:
                        x_hit = x + t * math.cos(angle)
                        if -room_size <= x_hit <= room_size:
                            if t < min_dist:
                                min_dist = t
        
        return min_dist if min_dist < max_range else float('inf')
    
    def publish_sensor_data(self):
        """Публикация данных с обоих датчиков"""
        # Левый датчик (смотрит немного влево)
        left_dist = self.calculate_distance_to_obstacle(math.radians(15))
        self.publish_range(self.left_pub, 'left', left_dist)
        
        # Правый датчик (смотрит немного вправо)
        right_dist = self.calculate_distance_to_obstacle(math.radians(-15))
        self.publish_range(self.right_pub, 'right', right_dist)
        
    def publish_range(self, publisher, sensor_name, distance):
        """Формирование и публикация сообщения Range"""
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'tof_{sensor_name}_link'
        
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = self.get_parameter('fov').value
        msg.min_range = self.get_parameter('min_range').value
        msg.max_range = self.get_parameter('max_range').value
        
        if math.isfinite(distance):
            msg.range = distance
        else:
            msg.range = float('inf')  # нет препятствия в пределах дальности
            
        publisher.publish(msg)
        
        # Логирование для отладки (можно отключить)
        # self.get_logger().info(f'{sensor_name}: {msg.range:.3f} м')

def main(args=None):
    rclpy.init(args=args)
    node = TofSimulator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
