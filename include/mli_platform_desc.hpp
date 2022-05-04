#ifndef _MLI_PLATFORM_DESC_HPP_
#define _MLI_PLATFORM_DESC_HPP_

#include <stdint.h>
#include "mli_debug.h"

namespace snps_arc::metaware::mli {

class PlatformDescription {	
public:
    enum AguConfig
    {
        kAguConfigSmall,
	kAguConfigMedium,
	kAguConfigLarge,
	kAguConfigNoAgu
    };
    enum RoundingMode
    {
        kRoundingModeConvergent,
	kRoundingModeUp
    };


    PlatformDescription()
    : m_guard_bits(0)
    , m_vector_length_8bit(1)
    , m_vector_length_16bit(1)
    , m_mac_issue_slots (1)
    , m_rounding_mode(kRoundingModeConvergent)
    , m_processor_id (0)
    , m_agu_config(kAguConfigNoAgu)
    {}
	
	
	
    uint32_t GetGuardBits() const { return m_guard_bits; }
	
    uint32_t GetVectorLength8bit() const { return m_vector_length_8bit; }
	
    uint32_t GetVectorLength16bit() const { return m_vector_length_16bit; }
	
    uint32_t GetMacIssueSlots() const { return m_mac_issue_slots; }
	
    uint32_t GetProcessorId() const { return m_processor_id; }
	
    RoundingMode GetRoundingMode() const { return m_rounding_mode; }
	
    AguConfig GetAguConfig() const { return m_agu_config; }
	
	
	
    void SetGuardBits(uint32_t guard_bits){
        MLI_ASSERT(guard_bits == 0 || guard_bits == 4 || guard_bits == 8);
        m_guard_bits = guard_bits;
    }
	
	
    void SetVectorLength8bit(uint32_t vector_length_8bit){
        MLI_ASSERT(vector_length_8bit == 1
        || vector_length_8bit == 2
        || vector_length_8bit == 64
        || vector_length_8bit == 32
        || vector_length_8bit == 16
        );
        m_vector_length_8bit = vector_length_8bit;
    }
	
	
    void SetVectorLength16bit(uint32_t vector_length_16bit){
        MLI_ASSERT(vector_length_16bit == 1
        || vector_length_16bit == 2
        || vector_length_16bit == 32
        || vector_length_16bit == 16
        || vector_length_16bit == 8
        );
        m_vector_length_16bit = vector_length_16bit;
    }
	
	
    void SetMacIssueSlots(uint32_t mac_issue_slots){
        MLI_ASSERT( mac_issue_slots == 1 || mac_issue_slots == 2);
        m_mac_issue_slots = mac_issue_slots;
    }
	
	
    void SetProcessorId(uint32_t processor_id){ m_processor_id = processor_id; }
	
	
    void SetRoundingMode(RoundingMode rounding_mode){
        MLI_ASSERT(rounding_mode == kRoundingModeConvergent || rounding_mode == kRoundingModeUp);
        m_rounding_mode = rounding_mode;
    }
	
	
    void SetAguConfig(AguConfig agu_config){
        MLI_ASSERT(agu_config == kAguConfigSmall || agu_config == kAguConfigMedium || agu_config == kAguConfigLarge || agu_config == kAguConfigNoAgu);
        m_agu_config = agu_config;
    }
	

private:
    uint32_t m_guard_bits;
    uint32_t m_vector_length_8bit;
    uint32_t m_vector_length_16bit;
    uint32_t m_mac_issue_slots;
    RoundingMode m_rounding_mode;
    uint32_t m_processor_id;
    AguConfig m_agu_config;
};
}

#endif
