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

#include "configurator/motor_param.hpp"

MotorParamConfigurator::MotorParamConfigurator(roboteq::serial_controller *serial, std::string name, unsigned int number)
    : mSerial(serial)
{
    // Find path param
    mName =  name;
    // Roboteq motor number
    mNumber = number;
    // Set false on first run
    setup_param = false;
}
//replaced
/*void MotorParamConfigurator::initConfigurator(bool load_from_board, std::map<std::string, std::any>& roboteq_params)
{
    double ratio;
    // if ratio variable does not exists, set it
    if(roboteq_params.find(mName+"_ratio") == roboteq_params.end())
    {
        roboteq_params[mName + "_ratio"] = 1.0;
        RCLCPP_INFO(logger_, "Ratio set to default: %d", roboteq_params[mName + "_ratio"]);
        
    } 

    if(load_from_board)
    {
        // Load parameters from roboteq
        getParamFromRoboteq(roboteq_params);
    }
}*/
//added
void MotorParamConfigurator::initConfigurator(
    bool load_from_board,
    std::map<std::string, std::any>& roboteq_params)
{
    // Set the default ratio if it is not already configured.
    if (roboteq_params.find(mName + "_ratio") == roboteq_params.end())
    {
        const double default_ratio = 1.0;
        roboteq_params[mName + "_ratio"] = default_ratio;

        RCLCPP_INFO(
            logger_,
            "Ratio set to default: %f",
            default_ratio
        );
    }

    if (load_from_board)
    {
        getParamFromRoboteq(roboteq_params);
    }
}
//until here

void MotorParamConfigurator::getParamFromRoboteq(std::map<std::string, std::any>& roboteq_params){
    try{
        //replaced
        //double ratio = std::any_cast<int>(roboteq_params[mName+"_ratio"]);
        //added
        const double ratio =
            std::any_cast<double>(roboteq_params.at(mName + "_ratio"));
        //until here
        // Motor direction {1 (Clockwise), -1 (Underclockwise)}
        string str_mdir = mSerial->getParam("MDIR", std::to_string(mNumber));
        // Get sign from roboteq board
        int sign = boost::lexical_cast<int>(str_mdir) ? -1 : 1;
        // Set parameter
        roboteq_params[mName+"_rotation"] = sign;

        // Stall detection
        string str_stall = mSerial->getParam("BLSTD", std::to_string(mNumber));
        int stall = boost::lexical_cast<int>(str_stall);
        // Set params
        roboteq_params[mName+"_stallDetection"] = stall;

        // Get Max Amper limit = alim / 10
        string str_alim = mSerial->getParam("ALIM", std::to_string(mNumber));
        unsigned int tmp = boost::lexical_cast<unsigned int>(str_alim);
        double alim = ((double) tmp) / 10.0;
        // Set params
        roboteq_params[mName+"_amperLimit"] = alim;

        // Max power forward
        string str_max_fw = mSerial->getParam("MXPF", std::to_string(mNumber));
        // Get max forward
        int max_forward = boost::lexical_cast<unsigned int>(str_max_fw);
        // Set parameter
        roboteq_params[mName+"_maxAcceleration"] = max_forward; //should it be _maxForward ??

        // Max power forward reverse
        string str_max_re = mSerial->getParam("MXPR", std::to_string(mNumber));
        // Get max reverse
        int max_reverse = boost::lexical_cast<unsigned int>(str_max_re);
        // Set parameter
        roboteq_params[mName+"_maxDeceleration"] = max_reverse; //should it be _maxReverse ??

        // Get Max RPM motor
        string str_rpm_motor = mSerial->getParam("MXRPM", std::to_string(mNumber));
        // Get RPM from board
        unsigned int rpm_motor = boost::lexical_cast<unsigned int>(str_rpm_motor);
        // Convert in max RPM
        double max_rpm = ((double) rpm_motor) / ratio;
        // Set parameter
        roboteq_params[mName+"_maxSpeed"] = max_rpm;
        RCLCPP_INFO(logger_, "Setting max_Speed in configurator: %f",std::any_cast<double>(roboteq_params[mName+"_maxSpeed"]));
        // Get Max RPM acceleration rate
        string str_rpm_acceleration_motor = mSerial->getParam("MAC", std::to_string(mNumber));
        // Get RPM from board
        unsigned int rpm_acceleration_motor = boost::lexical_cast<unsigned int>(str_rpm_acceleration_motor);
        // Convert in max RPM
        double rpm_acceleration = ((double) rpm_acceleration_motor) / ratio;
        // Set parameter
        roboteq_params[mName+"_maxAcceleration"] = rpm_acceleration;

        // Get Max RPM deceleration rate
        string str_rpm_deceleration_motor = mSerial->getParam("MDEC", std::to_string(mNumber));
        // Get RPM from board
        unsigned int rpm_deceleration_motor = boost::lexical_cast<unsigned int>(str_rpm_deceleration_motor);
        // Convert in max RPM
        double rpm_deceleration = ((double) rpm_deceleration_motor) / ratio;
        // Set parameter
        roboteq_params[mName+"_maxDeceleration"] = rpm_deceleration;


    } catch (std::bad_cast& e){
        RCLCPP_ERROR(logger_, "Failure parsing feedback data. Dropping message. %s", e.what());
    }
}

void MotorParamConfigurator::setOperativeMode(int type)
{
    // Update operative mode
    mSerial->setParam("MMOD", std::to_string(mNumber) + " " + std::to_string(type));
}

int MotorParamConfigurator::getOperativeMode()
{
    // Operative mode reference in [pag 321]
    string str_mode = mSerial->getParam("MMOD", std::to_string(mNumber));
    // Get sign from roboteq board
    int mode = boost::lexical_cast<int>(str_mode);
    return mode;
}

