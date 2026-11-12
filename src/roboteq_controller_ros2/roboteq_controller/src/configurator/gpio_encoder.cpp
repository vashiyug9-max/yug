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

#include "configurator/gpio_encoder.hpp"

#define PARAM_ENCODER_STRING "_encoder"

GPIOEncoderConfigurator::GPIOEncoderConfigurator(roboteq::serial_controller *serial, std::vector<roboteq::Motor *> motor, string name, unsigned int number)
    : mSerial(serial)
    ,_motor(motor)
{
    // Find path param
    mName = name + PARAM_ENCODER_STRING + "_" + std::to_string(number);
    // Roboteq motor number
    mNumber = number;
    // Set false on first run
    setup_encoder = false;
}

void GPIOEncoderConfigurator::initConfigurator(bool load_from_board, std::map<std::string, std::any>& roboteq_params)
{
    // Get PPR Encoder parameter
    double ppr;
    ppr = std::any_cast<int>(roboteq_params[mName + "_PPR"]);
    _reduction = ppr;
    // Multiply for quadrature
    _reduction *= 4;

    // Check if is required load paramers
    if(load_from_board)
    {
        // Load encoder properties from roboteq
        getEncoderFromRoboteq(roboteq_params);
    }

}

double GPIOEncoderConfigurator::getConversion(double reduction, std::map<std::string, std::any>& roboteq_params) {
    
    if(roboteq_params.find(mName+"_position")!=roboteq_params.end()){
        if(std::any_cast<int>(roboteq_params[mName+"_position"]))
            {
                RCLCPP_INFO(logger_, "_reduction read from encoder(PPR*4): %f. Multiplying it by the reduction from yaml: %f  ", _reduction, reduction);
                return _reduction * reduction;
            }
    }

    return _reduction;
}

void GPIOEncoderConfigurator::getEncoderFromRoboteq(std::map<std::string, std::any>& roboteq_params) {
    try
    {
        // Get Encoder Usage - reference
        string str_emode = mSerial->getParam("EMOD", std::to_string(mNumber));
        // Get PPR from roboteq board
        int emod = boost::lexical_cast<unsigned int>(str_emode);
        // 3 modes:
        // 0 - Unsed
        // 1 - Command
        // 2 - Feedback
        int command = (emod & 0b11);
        int motors = (emod - command) >> 4;
        int tmp1 = ((motors & 0b1) > 0);
        int tmp2 = ((motors & 0b10) > 0);
        // Register reduction
        if(tmp1)
        {
            for (vector<roboteq::Motor*>::iterator it = _motor.begin() ; it != _motor.end(); ++it)
            {
                roboteq::Motor* motor = ((roboteq::Motor*)(*it));
                if(motor->getNumber() == 1)
                {
                    motor->registerSensor(this, roboteq_params);
                    RCLCPP_INFO(logger_,"Register encoder [%d] to: %s",mNumber, motor->getName());
                    break;
                }
            }
        }
        if(tmp2)
        {
            for (vector<roboteq::Motor*>::iterator it = _motor.begin() ; it != _motor.end(); ++it)
            {
                roboteq::Motor* motor = ((roboteq::Motor*)(*it));
                if(motor->getNumber() == 2)
                {
                    motor->registerSensor(this, roboteq_params);
                    RCLCPP_INFO(logger_, "Register encoder [%d] to: %s",mNumber, motor->getName());
                    break;
                }
            }
        }
        // Set parameter
        roboteq_params[mName + "_configuration"] = command;
        roboteq_params[mName + "_inputMotorOne"] = tmp1;
        roboteq_params[mName + "_inputMotorTwo"] = tmp2;

        // Get Encoder PPR (Pulse/rev)
        string str_ppr = mSerial->getParam("EPPR", std::to_string(mNumber));
        // Get PPR from roboteq board
        int ppr = boost::lexical_cast<unsigned int>(str_ppr);
        RCLCPP_INFO(logger_, "Motor: %s , PPR: %d", mName.c_str(), ppr);
        // Set parameter
        roboteq_params[mName + "_PPR"] = ppr;


        // Get Encoder ELL - Min limit
        string str_ell = mSerial->getParam("ELL", std::to_string(mNumber));
        // Get PPR from roboteq board
        int ell = boost::lexical_cast<unsigned int>(str_ell);
        // Set parameter
        roboteq_params[mName + "_encoderLowCountLimit"] = ell;

        // Get Encoder EHL - Max limit
        string str_ehl = mSerial->getParam("EHL", std::to_string(mNumber));
        // Get PPR from roboteq board
        int ehl = boost::lexical_cast<unsigned int>(str_ehl);
        // Set parameter
        roboteq_params[mName + "_encoderHighCountLimit"] = ehl;

        // Get Encoder EHOME - Home count
        string str_home = mSerial->getParam("EHOME", std::to_string(mNumber));
        // Get PPR from roboteq board
        int home = boost::lexical_cast<unsigned int>(str_home);
        // Set parameter
        roboteq_params[mName + "_encoderHomeCount"] = home;


    } catch (std::bad_cast& e)
    {
        RCLCPP_WARN(logger_, "Failure parsing feedback data. Dropping message.%s", e.what());
    }
}
