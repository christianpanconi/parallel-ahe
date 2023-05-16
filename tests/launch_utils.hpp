#ifndef LAUNCH_UTILS_HPP_
#define LAUNCH_UTILS_HPP_

#include <string>
#include <vector>

namespace EqArgs{
	static const std::string EQ_TYPE_BI="bi";
	static const std::string EQ_TYPE_MONO="mono";
	const char *usage = R"(Equalization args:
-f  | --file              <file path>      Input image.
-e  | --equalization-type <"mono"|"bi">    Equalization type, either "mono" or "bi".
-ws | --window-size       <int>            Window size (if even will be set to
                                           the next odd integer).
)";

	// configs
	typedef struct Configs{
		std::string equalization_type = EqArgs::EQ_TYPE_BI;
		std::string img_file = "img/squared/ducks_1000.ppm";

		unsigned int window_size = 31;
		std::string to_string(){
			std::string str("");
			str += "\n\tequalization type: " + equalization_type +
				   "\n\timage:             " + img_file +
				   "\n\twindow size:       " + std::to_string(window_size);
			return str;
		}
	} Configs;

	// functions
	EqArgs::Configs parseCmdEqualizationArgs( int argc , char** argv ){
		std::vector<char *> args;
		std::copy(&argv[0] , &argv[argc] , std::back_inserter(args) );
		EqArgs::Configs eargs;

		int i=0;
		std::string arg;
		while( i < args.size() ){
			arg = args[i];
			if( arg == "-f" || arg == "--file" ){
				eargs.img_file=argv[i+1]; i++;
			}
			if( arg == "-e" || arg == "--equalization-type" ){
				eargs.equalization_type=argv[i+1]; i++;
			}
			if( arg == "-ws" || arg == "--window-size" ){
				eargs.window_size=std::stoi(argv[i+1]); i++;
			}
			i++;
		}

		return eargs;
	}

}

#endif /* LAUNCH_UTILS_HPP_ */
