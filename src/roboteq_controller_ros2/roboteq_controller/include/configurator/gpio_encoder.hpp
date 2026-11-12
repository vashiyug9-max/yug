/**
 * Copyright (C) 2017, Raffaello Bonghi <raffaello@rnext.it>
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GPIOENCODERCONFIGURATOR_H
#define GPIOENCODERCONFIGURATOR_H

#include <rclcpp/rclcpp.hpp>

#include "roboteq_controller/serial_controller.hpp"

#include "roboteq_controller/motor.hpp"
#include "configurator/gpio_sensor.hpp"

class GPIOEncoderConfigurator : GPIOSensor
{
public:
    /**
     * @brief GPIOEncoderConfigurator
     * @param nh
     * @param serial
     * @param number
     */
    GPIOEncoderConfigurator(roboteq::serial_controller *serial, std::vector<roboteq::Motor *> motor, string name, unsigned int number);
    /**
     * @brief initConfigurator Initialize all parameter and syncronize parameters between ros and roboteq board
     * @param load_from_board If true load all paramter from roboteq board
     */
    void initConfigurator(bool load_from_board, std::map<std::string, std::any>& roboteq_params);
    /**
     * @brief getConversion Get conversion from pulse value to real value
     * @return the value of reduction before encoder
     */
    double getConversion(double reduction,std::map<std::string, std::any>& roboteq_params);

private:
    /// Setup variable
    bool setup_encoder;

    /// Associate name space
    string mName;
    /// Number motor
    unsigned int mNumber;

    /// Serial port
    roboteq::serial_controller* mSerial;
    // List of all motors
    std::vector<roboteq::Motor *> _motor;

    // reduction value
    double _reduction;

    /**
     * @brief getEncoderFromRoboteq Load Encoder parameters from Roboteq board
     */
    void getEncoderFromRoboteq(std::map<std::string, std::any>& roboteq_params);
    rclcpp::Logger logger_ = rclcpp::get_logger("ros2_control_node");
};

#endif // GPIOENCODERCONFIGURATOR_H
