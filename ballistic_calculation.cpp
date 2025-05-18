#include <iostream>
#include <cmath>

const double g = 9.8; // 重力加速度(m/s^2)
const double PI = 3.141592653589793;
const double rho = 1.225; // 空气密度(kg/m^3)
const double Cd = 0.47; // 阻力系数(球体)
const double r = 0.04; // 弹体半径(m)

/**
 * 计算发射角度的函数
 * @param m 质量(kg)
 * @param v 初速度(m/s)
 * @param d 目标水平距离(m)
 * @return 发射角度(弧度)
 */
double calculateLaunchAngle(double m, double v, double d) {
    // 考虑空气阻力的弹道计算
    // 斯托克斯定律计算阻力
    double A = PI * r * r; // 横截面积
    
    // 使用二分法寻找最优角度
    double low = 0.0;
    double high = PI/2;
    double best_theta = 0.0;
    double min_error = 1e10;
    
    for(int i = 0; i < 20; i++) {
        double theta = (low + high) / 2;
        double t = 0.0;
        double dt = 0.01;
        double x = 0.0, y = 0.0;
        double vx = v * cos(theta);
        double vy = v * sin(theta);
        
        while (y >= 0) {
            double speed = sqrt(vx*vx + vy*vy);
            double drag = 0.5 * Cd * rho * A * speed * speed;
            
            // 四阶龙格库塔法
            double k1vx = -drag * vx / speed / m;
            double k1vy = -g - drag * vy / speed / m;
            
            double k2vx = -drag * (vx + 0.5*dt*k1vx) / sqrt((vx+0.5*dt*k1vx)*(vx+0.5*dt*k1vx)+(vy+0.5*dt*k1vy)*(vy+0.5*dt*k1vy)) / m;
            double k2vy = -g - drag * (vy + 0.5*dt*k1vy) / sqrt((vx+0.5*dt*k1vx)*(vx+0.5*dt*k1vx)+(vy+0.5*dt*k1vy)*(vy+0.5*dt*k1vy)) / m;
            
            double k3vx = -drag * (vx + 0.5*dt*k2vx) / sqrt((vx+0.5*dt*k2vx)*(vx+0.5*dt*k2vx)+(vy+0.5*dt*k2vy)*(vy+0.5*dt*k2vy)) / m;
            double k3vy = -g - drag * (vy + 0.5*dt*k2vy) / sqrt((vx+0.5*dt*k2vx)*(vx+0.5*dt*k2vx)+(vy+0.5*dt*k2vy)*(vy+0.5*dt*k2vy)) / m;
            
            double k4vx = -drag * (vx + dt*k3vx) / sqrt((vx+dt*k3vx)*(vx+dt*k3vx)+(vy+dt*k3vy)*(vy+dt*k3vy)) / m;
            double k4vy = -g - drag * (vy + dt*k3vy) / sqrt((vx+dt*k3vx)*(vx+dt*k3vx)+(vy+dt*k3vy)*(vy+dt*k3vy)) / m;
            
            vx += dt * (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6;
            vy += dt * (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6;
            
            x += vx * dt;
            y += vy * dt;
            t += dt;
        }
        
        double error = x - d;
        if(fabs(error) < min_error) {
            min_error = fabs(error);
            best_theta = theta;
        }
        
        if(error > 0) {
            high = theta;
        } else {
            low = theta;
        }
    }
    
    return best_theta;
}

int main() {
    double mass, velocity, distance;
    
    std::cout << "请输入质量(kg): ";
    std::cin >> mass;
    
    std::cout << "请输入初速度(m/s): ";
    std::cin >> velocity;
    
    std::cout << "请输入目标水平距离(m): ";
    std::cin >> distance;
    
    // 验证输入
    if (mass <= 0 || velocity <= 0 || distance <= 0) {
        std::cerr << "错误: 输入参数必须为正数" << std::endl;
        return -1;
    }
    
    double angle = calculateLaunchAngle(mass, velocity, distance);
    
    if (angle >= 0) {
        std::cout << "发射角度: " << angle * 180.0 / PI << " 度" << std::endl;
        
    }
    
    return 0;
}