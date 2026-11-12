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

#ifndef ROBOTEQ_H
#define ROBOTEQ_H

#include "rclcpp/rclcpp.hpp"
#include <serial/serial.h>
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"


#include "configurator/gpio_analog.hpp"
#include "configurator/gpio_pulse.hpp"
#include "configurator/gpio_encoder.hpp"

#include "diagnostic_updater/diagnostic_updater.hpp"
#include "diagnostic_updater/publisher.hpp"

#include "roboteq_controller/motor.hpp"
#include <map>
#include <any>



using namespace std;
#define hardwareConnected 1

namespace roboteq
{

typedef struct joint
{
    Motor *motor;
    // State of the motor
    double position;
    double velocity;
    double effort;
    double velocity_command;
} joint;


typedef struct _status_flag {
    uint8_t serial_mode : 1;
    uint8_t pulse_mode : 1;
    uint8_t analog_mode : 1; 
    uint8_t spectrum : 1;
    uint8_t power_stage_off : 1;
    uint8_t stall_detect : 1;
    uint8_t at_limit : 1;
    uint8_t microbasic_running : 1;
} status_flag_t;
// Reference in pag 245
typedef struct _status_fault {
    uint8_t overheat : 1; 
    uint8_t overvoltage : 1;
    uint8_t undervoltage : 1;
    uint8_t short_circuit : 1;
    uint8_t emergency_stop : 1;
    uint8_t brushless_sensor_fault : 1;
    uint8_t mosfet_failure : 1;
} status_fault_t;

std::vector<string> inout_channel_input_type{"analog","pulse"};
std::vector<string> channel_config_params_name{"conversion","conversionPolarity","inputDeadband", "inputMotorOne","inputMotorTwo", "inputUse", "rangeInputCenter", "rangeInputMax","rangeInputMin"};
std::vector<string> encoder_params_name{"PPR", "configuration", "encoderHighCountLimit", "encoderHomeCount", "encoderLowCountLimit", "inputMotorOne", "inputMotorTwo", "position"};
std::vector<string> pid_params_name{"Kp","Ki","Kd","integratorLimit", "loopErrorDetection", "positionModeVelocity", "turnMinToMax"};
std::vector<string> joint_params_name{"number", "ratio", "rotation", "stallDetection", "amperLimit", "maxSpeed", "maxAcceleration", "maxDeceleration", "maxForward", "maxReverse", "pid_closedLoopPosition", "pid_closedLoopVelocity", "pid_position", "pid_torque", "pid_velocity" };



class Roboteq : public hardware_interface::SystemInterface //, public diagnostic_updater::DiagnosticTask
{
public:
    /**
     * @brief Roboteq The Roboteq board controller write and read messages about the motor state
     * @param serial The serial controller
     */
    //Roboteq(const ros::NodeHandle &nh, const ros::NodeHandle &private_nh, serial_controller *serial);
    /**
      * @brief The deconstructor
      */
    Roboteq();  // pluginlib dynamically creates a plugin instace using a default constructor. Hence it needs to be defined 
    ~Roboteq();
    /**
     * @brief Switch off roboteq board
     */

    RCLCPP_SHARED_PTR_DEFINITIONS(Roboteq)  //allows to create shared ptr for the class

    rclcpp::Logger logger_ = rclcpp::get_logger("ros2_control_node");

    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
    hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
    hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
    
    std::vector<YAML::Node> motor_config;       //0-> left_motor, 1-> right_motor

    std::map<std::string, std::any> roboteq_params;
    std::vector<double> hw_state_pos_, hw_state_vel_, hw_command_;

    void getParams(const hardware_interface::HardwareInfo & info_);
    void setParam(const hardware_interface::HardwareInfo & info_, string name);
    void switch_off();

private:

    // Serial controller
    serial_controller *mSerial;
    /// URDF information about robot
    urdf::Model model;

    bool motor_loop_;
    // Check if is the first run
    bool _first;
    // Motor definition
    std::vector<Motor*> mMotor;

    string _type, _model;
    string _version;
    string _uid;
    // Status Roboteq board
    status_flag_t _flag;
    // Fault flags Roboteq board
    status_fault_t _fault;
    // Volts internal
    double _volts_internal, _volts_five;
    // Tempearture inside the Roboteq board
    double _temp_mcu, _temp_bridge;

    // GPIO enable read
    bool _isGPIOreading;

    std::vector<GPIOAnalogConfigurator*> _param_analog;
    std::vector<GPIOPulseConfigurator*> _param_pulse;
    // Encoder
    std::vector<GPIOEncoderConfigurator*> _param_encoder;

    uint8_t motorDiagnosticCallCounter = 0;

    /**
     * @brief getRoboteqInformation Load basic information from roboteq board
     */
    void getRoboteqInformation();
    void initialize();
    void motorDiagnostic();
    void getControllerFromRoboteq();

    /// Setup variable
    bool setup_controller;

};

}
#endif // ROBOTEQ_H
