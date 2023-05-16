/*
 * timer.hpp
 */
#ifndef C8_TIMER_H_
#define C8_TIMER_H_

#include <chrono>
#include <stdio.h>

namespace c8{

class Timer {
private:
	bool started;
	bool stopped;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_end;

	// Trait to check if the template parameter is a std::chrono::duration
	template <typename T>
	struct is_chrono_duration
	{ static constexpr bool value = false; };

	template<typename Rep , typename Period>
	struct is_chrono_duration<std::chrono::duration<Rep , Period>>
	{ static constexpr bool value = true; };



public:
	Timer(){
		this->started = false;
		this->stopped = false;
	};

	void start(){
		if( !started ){
			this->t_start = std::chrono::high_resolution_clock::now();
			started = true;
			stopped = false;
		}
	};

	void stop(){
		if( started ){
			this->t_end = std::chrono::high_resolution_clock::now();
			started = false;
			stopped = true;
		}
	};

	template<typename Precision> unsigned long long int elapsed(){
		static_assert( is_chrono_duration<Precision>::value , "Wrong template parameter type. std::chrono::duration required." );

		if( started || !stopped ){
			printf( "The timer is still running or it has not been STARTED and STOPPED.\n" );
			throw "The timer is still running or it has not been STARTED and STOPPED.";
		}

		return (unsigned long long int) std::chrono::duration_cast<Precision>
			(this->t_end - this->t_start).count();
	}


};

}

#endif /* C8_TIMER_H_ */
