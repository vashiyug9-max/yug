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

#include "configurator/motor_pid.hpp"

MotorPIDConfigurator::MotorPIDConfigurator(roboteq::serial_controller *serial, string name, string type, unsigned int number)
    : mSerial(serial)
{
    // Find path param
    mName = name;
    mType = type;
    RCLCPP_DEBUG_STREAM(logger_, "Param " << name + "_pid_" + type << " N:" << number);
    // Roboteq motor number
    mNumber = number;
    // Set false on first run
    setup_pid = false;

}

void MotorPIDConfigurator::initConfigurator(bool load_from_board, std::map<std::string, std::any>& roboteq_params)
{
    // Check if is required load paramers
    if(load_from_board)
    {
        // Load parameters from roboteq
        getPIDFromRoboteq(roboteq_params);
    }
}

void MotorPIDConfigurator::getPIDFromRoboteq(std::map<std::string, std::any>& roboteq_params)
{
    try
    {
        // Get Position velocity [pag. 322]
        string str_pos_vel = mSerial->getParam("MVEL", std::to_string(mNumber));
        int pos_vel = boost::lexical_cast<unsigned int>(str_pos_vel);
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_positionModeVelocity"] = pos_vel;


        // Get number of turn between limits [pag. 325]
        string str_mxtrn = mSerial->getParam("MXTRN", std::to_string(mNumber));
        unsigned int tmp_mxtrn = boost::lexical_cast<unsigned int>(str_mxtrn);
        double mxtrn = ((double) tmp_mxtrn) / 100.0;
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_turnMinToMax"] = mxtrn;

        // Get KP gain = kp / 10 [pag 319]
        string str_kp = mSerial->getParam("KP", std::to_string(mNumber));
        unsigned int tmp_kp = boost::lexical_cast<unsigned int>(str_kp);
        double kp = ((double) tmp_kp) / 10.0;
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_Kp"] = kp;

        // Get KI gain = ki / 10 [pag 318]
        string str_ki = mSerial->getParam("KI", std::to_string(mNumber));
        unsigned int tmp_ki = boost::lexical_cast<unsigned int>(str_ki);
        double ki = ((double) tmp_ki) / 10.0;
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_Ki"] = ki;

        // Get KD gain = kd / 10 [pag 317]
        string str_kd = mSerial->getParam("KD", std::to_string(mNumber));
        unsigned int tmp_kd = boost::lexical_cast<unsigned int>(str_kd);
        double kd = ((double) tmp_kd) / 10.0;
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_Kd"] = kd;

        // Get Integral cap [pag. 317]
        string str_icap = mSerial->getParam("ICAP", std::to_string(mNumber));
        int icap = boost::lexical_cast<unsigned int>(str_icap);
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_integratorLimit"] = icap;

        // Get closed loop error detection [pag. 311]
        string str_clred = mSerial->getParam("CLERD", std::to_string(mNumber));
        int clerd = boost::lexical_cast<unsigned int>(str_clred);
        // Set params
        roboteq_params[mName + "_pid_" + mType + "_loopErrorDetection"] = clerd;

    } catch (std::bad_cast& e)
    {
        RCLCPP_DEBUG(logger_, "Failure parsing feedback data. Dropping message. %s", e.what());
    }

}

//function  never called
void MotorPIDConfigurator::setPIDconfiguration(std::map<std::string, std::any>& roboteq_params)
{
    // Set Position velocity
    int pos_vel;
    // Set params
    pos_vel = std::any_cast<int>(roboteq_params[mName + "_pid_" + mType + "_positionModeVelocity"]);
    // Update position velocity
    mSerial->setParam("MVEL", std::to_string(mNumber) + " " + std::to_string(pos_vel));

    // Set number of turn between limits
    double mxtrn = std::any_cast<double>(roboteq_params[mName + "_pid_" + mType + "_turnMinToMax"]);
    // Update position velocity
    mSerial->setParam("MXTRN", std::to_string(mNumber) + " " + std::to_string(mxtrn * 100));

    // Set KP gain = kp * 10
    double kp = std::any_cast<double>(roboteq_params[mName + "_pid_" + mType + "_Kp"]);
    // Update gain position
    mSerial->setParam("KP", std::to_string(mNumber) + " " + std::to_string(kp * 10));

    // Set KI gain = ki * 10
    double ki = std::any_cast<double>(roboteq_params[mName + "_pid_" + mType + "_Ki"]);
    // Set KI parameter
    mSerial->setParam("KI", std::to_string(mNumber) + " " + std::to_string(ki * 10));

    // Set KD gain = kd * 10
    double kd = std::any_cast<double>(roboteq_params[mName + "_pid_" + mType + "_Kd"]);
    // Set KD parameter
    mSerial->setParam("KD", std::to_string(mNumber) + " " + std::to_string(kd * 10));

    // Set Integral cap
    int icap = std::any_cast<int>(roboteq_params[mName + "_pid_" + mType + "_integratorLimit"]);
    // Update integral cap
    mSerial->setParam("ICAP", std::to_string(mNumber) + " " + std::to_string(icap));

    // Set closed loop error detection
    int clerd = std::any_cast<int>(roboteq_params[mName + "_pid_" + mType + "_loopErrorDetection"]);
    // Update integral cap
    mSerial->setParam("CLERD", std::to_string(mNumber) + " " + std::to_string(clerd));
}

