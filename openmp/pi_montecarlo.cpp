#include <omp.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

std::vector<uint32_t> get_seeds(size_t count) {
	std::vector<uint32_t> seeds(count);
    	size_t thread_id;
	size_t seed;
    	#pragma omp parallel for private(thread_id, seed) 
        for(int i=0;i<count;++i){
        	thread_id = omp_get_thread_num();
        	seed = (unsigned int) time(NULL);
        	seeds[thread_id] = (seed & 0xFFFFFFF0) | (thread_id + 1);
    	}
	return seeds;
    }

double pi_montecarlo(size_t num_threads, long long N) {
    	long long points_incircle = 0;
    	size_t thread_id;
    	omp_set_num_threads(num_threads);
	std::vector<uint32_t> seeds = get_seeds(num_threads);
   	uint32_t seed;
    	double start, end;
    	start = omp_get_wtime();

	#pragma omp parallel shared(seeds) \
        private(thread_id, seed) reduction(+:points_incircle) 
    	{
        	thread_id = omp_get_thread_num();
        	seed = seeds[thread_id];
        	#pragma omp for
        	for(int i = 0; i < N; i++) {
            		double x = ((double) rand_r(&seed) / (RAND_MAX));
            		double y = ((double) rand_r(&seed) / (RAND_MAX));
            		if (x*x + y*y < 1 + 1e-14) {
                		points_incircle += 1;
            		}
        	}	
    	}
    	end = omp_get_wtime();
    	printf("Time is %lf\n", end - start);
    	return 4.0*points_incircle/N;
}

int main(int argc, char ** argv){
    	size_t num_threads = 1;
    	long long N = 10000;
    	if (argc > 1) {
        	num_threads = std::stoi(argv[1]);
    	}
    	if (argc > 2) {
        	N = std::stoll(argv[2]);
    	}
    	double pi = pi_montecarlo(num_threads, N);
    	printf("Pi = %.10lf\n", pi);
    	return 0;
}


