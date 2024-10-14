/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's portFFT
 *
 **************************************************************************/

#ifndef PORTFFT_SPECIALIZATION_CONSTANT_HPP
#define PORTFFT_SPECIALIZATION_CONSTANT_HPP

#include <sycl/sycl.hpp>

#include "defines.hpp"
#include "enums.hpp"
#include "common/logging.hpp"

namespace portfft::detail {

template <typename T>
struct shared_spec_constants {
// Specialization constants used for IFFT, when expressed as a IFFT=(conjugate(FFT(conjugate(input))))
  bool apply_multiply_on_load;
  bool apply_multiply_on_store;
  bool apply_conjugate_on_load;
  bool apply_conjugate_on_store;

  bool apply_scale_factor;

  complex_storage storage;
  T scale_factor;

  IdxGlobal input_stride;
  IdxGlobal output_stride;
  IdxGlobal input_distance;
  IdxGlobal output_distance;
};

constexpr static sycl::specialization_id<shared_spec_constants<float>> SpecConstSharedFloat{};
constexpr static sycl::specialization_id<shared_spec_constants<double>> SpecConstSharedDouble{};

template <typename T>
void set_shared_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, shared_spec_constants<T> constants){
    static_assert(std::is_trivially_copyable_v<shared_spec_constants<T>>());
    PORTFFT_LOG_TRACE("Spec Constant multiply_on_load:", constants.multiply_on_load);
    PORTFFT_LOG_TRACE("Spec Constant multiply_on_store:", constants.multiply_on_store);
    PORTFFT_LOG_TRACE("Spec Constant apply_scale_factor:", constants.apply_scale_factor);
    PORTFFT_LOG_TRACE("Spec Constant conjugate_on_load:", constants.conjugate_on_load);
    PORTFFT_LOG_TRACE("Spec Constant conjugate_on_store:", constants.conjugate_on_store);

    PORTFFT_LOG_TRACE("Spec Constant storage:", constants.storage);
    PORTFFT_LOG_TRACE("Spec Constant scaling factor:", constants.scaling_factor);

    PORTFFT_LOG_TRACE("Spec Constant input stride:", constants.input_stride);
    PORTFFT_LOG_TRACE("Spec Constant output stride:", constants.output_stride);
    PORTFFT_LOG_TRACE("Spec Constant input distance:", constants.input_distance);
    PORTFFT_LOG_TRACE("Spec Constant output distance:", constants.output_distance);
    if constexpr (std::is_same_v<T, float>) {
      in_bundle.template set_specialization_constant<SpecConstSharedFloat>(constants);
    } else {
      in_bundle.template set_specialization_constant<SpecConstSharedDouble>(constants);
    }
}

struct workitem_spec_constants {
  Idx fft_size;
};

constexpr static sycl::specialization_id<workitem_spec_constants> SpecConstWorkitem{};

static inline void set_workitem_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, workitem_spec_constants constants) {
    static_assert(std::is_trivially_copyable_v<workitem_spec_constants>);
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("Workitem Spec Constant fft_size:", constants.fft_size);
    in_bundle.template set_specialization_constant<SpecConstWorkitem>(constants);
}

struct subgroup_spec_constants {
    Idx factor_wi;
    Idx factor_sg;
};

constexpr static sycl::specialization_id<subgroup_spec_constants> SpecConstSubgroup{};

static inline void set_subgroup_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, subgroup_spec_constants constants) {
    static_assert(std::is_trivially_copyable_v<subgroup_spec_constants>);
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("Subgroup Spec Constant factor_wi:", constants.factor_wi);
    PORTFFT_LOG_TRACE("Subgroup Spec Constant factor_sg:", constants.factor_sg);
    in_bundle.template set_specialization_constant<SpecConstSubgroup>(constants);
}

struct workgroup_spec_constants {
  const Idx fft_size{};
};

constexpr static sycl::specialization_id<workgroup_spec_constants> SpecConstWorkgroup{};

void set_workgroup_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, workgroup_spec_constants constants) {
    static_assert(std::is_trivially_copyable_v<workgroup_spec_constants>);
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("Workgroup Spec Constant fft_size:", constants.fft_size);
    in_bundle.template set_specialization_constant<SpecConstWorkgroup>(constants);
}

struct global_spec_constants {
  detail::level level;
  Idx level_num;
  Idx num_factors;
};

union level_spec_constants {
  workitem_spec_constants workitem;
  subgroup_spec_constants subgroup;
  workgroup_spec_constants workgroup;
};

constexpr static sycl::specialization_id<global_spec_constants> SpecConstGlobal{};

void set_global_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, global_spec_constants constants, detail::level level, level_spec_constants level_constants) {
    static_assert(std::is_trivially_copyable_v<workgroup_spec_constants>);
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("global Spec Constant level:", constants.level);
    PORTFFT_LOG_TRACE("global Spec Constant level_num:", constants.level_num);
    PORTFFT_LOG_TRACE("global Spec Constant num_factors:", constants.num_factors);
    in_bundle.template set_specialization_constant<SpecConstGlobal>(constants);
    if (level == level::WORKITEM){
      set_workitem_spec_constants(in_bundle, level_constants.workitem);
    } else if (level == level::SUBGROUP){
      set_subgroup_spec_constants(in_bundle, level_constants.subgroup);
    } else if (level == level::WORKGROUP) {
      set_workgroup_spec_constants(in_bundle, level_constants.workgroup);
    }
}

struct transpose_spec_constants {
  portfft::complex_storage storage;
  Idx level;
  Idx num_factors;
};

constexpr static sycl::specialization_id<transpose_spec_constants> SpecConstTranspose{};

void set_transpose_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle, transpose_spec_constants constants) {
    static_assert(std::is_trivially_copyable_v<transpose_spec_constants>);
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("Transpose Spec Constant storage:", constants.storage);
    PORTFFT_LOG_TRACE("Transpose Spec Constant level:", constants.level);
    PORTFFT_LOG_TRACE("Transpose Spec Constant num_factors:", constants.num_factors);
    in_bundle.template set_specialization_constant<SpecConstTranspose>(constants);
}

}  // namespace portfft::detail
#endif
