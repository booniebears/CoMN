/**
 * @file Perf_Evaluator.h
 * @author booniebears
 * @brief Evaluate the performance info of the whole CIM arch.
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PERF_EVALUATOR_H_
#define PERF_EVALUATOR_H_

namespace Refactor {

// The interface of Calculating the Performance of all modules required.
// Include Macro,Buffer,Mesh,SFU,Htree etc.
void PPA_cost();

// Performance of buffer
void Buffer_Perf(int bufferSize, int buswidth, int featureSize);

// Performance of Orion(routers)
void Orion_Perf(int Fliter_size, int inPorts, int outPorts, int v_channels,
                double freq, int featureSize, bool isMesh);

} // namespace Refactor

#endif // !PERF_EVALUATOR_H_
