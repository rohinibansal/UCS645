CC=mpicc

all: q1 q2 q3 q4 q5

q1:
	$(CC) q1_daxpy.c -o q1

q2:
	$(CC) q2_bcast.c -o q2

q3:
	$(CC) q3_dotproduct.c -o q3

q4:
	$(CC) q4_primes.c -o q4 -lm

q5:
	$(CC) q5_perfect.c -o q5

clean:
	rm -f q1 q2 q3 q4 q5
