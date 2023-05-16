#ifndef SRC_EQUALIZATION_CPU_EQUALIZATION_COMMON_HPP_
#define SRC_EQUALIZATION_CPU_EQUALIZATION_COMMON_HPP_

unsigned char clamp8bit( unsigned int x ){
	return (unsigned char)( x > 255 ? 255 : x );
}

#endif /* SRC_EQUALIZATION_CPU_EQUALIZATION_COMMON_HPP_ */
