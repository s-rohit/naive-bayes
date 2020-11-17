/**
 * @date  : 2020-11-17
 * @author: Rohit S <rohit2013107@iitgoa.ac.in>
 * Wrapper class for memory-mapped filed
 */

// NOTE1. This is to optimize file read operations -- not _necessary_ for this project.
// You can safely ignore this class, and achieve the same results with std::ifstream.
// Hence, I am not writing comments for this class in great detail.

// NOTE2: For this simple demo, the class definition is included in the header itself.
// Remember to split this into appropriate header and cpp files for larger projects.


#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

class MemoryFile{
public:
	// pointers to beginning, current, and end of the file
	char *begin = 0, *end = 0, *curr = 0;
	// constructor (setup)
	MemoryFile(const char *fname){
		if(!fname) throw "error: filename not specified.";
		// file descriptor
		int fd = open(fname, O_RDONLY);
		if(fd < 0) throw "error: file not found.";
		// file size
		struct stat sb;
		if(fstat(fd, &sb)==-1) throw "error: file cannot be accessed.";
		if(sb.st_size == 0) throw "error: file empty.";
		// memory map
		void* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0u);
		if(addr == MAP_FAILED) throw "error: file could not be read.";
		// kernel optimization
		madvise(addr, sb.st_size, MADV_SEQUENTIAL);
		// init pointers
		begin = (char*) addr;
		end   = begin + sb.st_size;
		curr  = begin;
	}
	// destructor (free memory)
	~MemoryFile(){
		if(begin>0 && end>begin) 
			munmap(begin, end-begin);
	}
};