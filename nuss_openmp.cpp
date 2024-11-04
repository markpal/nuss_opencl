#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <openacc.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define paired(a1, a2) \
(((a1) == 'A' && (a2) == 'U') || \
((a1) == 'U' && (a2) == 'A') || \
((a1) == 'G' && (a2) == 'C') || \
((a1) == 'C' && (a2) == 'G'))



int N = 30000;
const int bb = 32;

short C[64][bb][bb] = {-1};
short A_elements[64][bb][bb] = {0};
short B_elements[64][bb][bb] = {0};
//int C[bb][bb];
//int A_elements[bb][bb];
//int B_elements[bb][bb];

using namespace std;

int main(){



  string seq = "GUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAC";

 N += bb - N % bb;
 //N = seq.length();


 int n = N, i,j,k;

  char *seqq = new char[N+1];
  if(N>1) // no debug
   {
    char znaki[] = {'C', 'G', 'U', 'A'};
    srand(static_cast<unsigned short>(time(0)));

    for (short i = 0; i < N; i++) {
      seqq[i] = znaki[rand() % 4];  // Losowy wybór z zestawu 'C', 'G', 'U', 'A'
    }
   }
   cout << seqq << endl;
  std::strcpy(seqq, seq.c_str());          // Copy the string content   // use random data for given big N, comment this

  short* flatArray_S = new short[n * n];
  short* flatArray_S_CPU = new short[n * n];

  // Allocate 2D host array for CPU and GPU
  short** S = new short*[n];
  short** S_CPU = new short*[n];

  for(short i = 0; i < n; i++) {
    S[i] = &flatArray_S[i * n];
    S_CPU[i] = &flatArray_S_CPU[i * n];
  }
  // initialization
  for(i=0; i<N; i++) {
    for(j=0; j<N; j++){
      S[i][j] = -1;
      S_CPU[i][j] = -1;
    }
  }
  for(i=0; i<N; i++){
    S[i][i] = 0;
    S_CPU[i][i] = 0;
    if(i+1 < N) {
      S[i][i + 1] = 0;
      S[i+1][i] = 0;
      S_CPU[i][i+1] = 0;
      S_CPU[i+1][i] = 0;
    }
  }
  // -----------------------------


  double start_time = omp_get_wtime();


   for (int c0 = 0; c0 <= (N - 1)/bb; c0 += 1)  // serial loop
   #pragma omp parallel for shared(c0)
        for (int c1 = c0; c1 <= min((N - 1) / bb, (N + c0 - 2 )/ bb); c1 += 1) // parallel loop  blocks
        {
               int id = omp_get_thread_num();


            for(int row=0; row<bb; row++)
                for(int col=0; col<bb; col++){
                  C[id][row][col] = -1;

                  }

            short _sj = c1-c0;
            short _si = c1;

            for (short m = _sj+1; m < _si; m++) {

                for(int row=0; row<bb; row++)
                    for(int col=0; col<bb; col++){
                        A_elements[id][row][col] = S[bb * _sj + row][bb * m  + col - 1];
                        B_elements[id][row][col] = S[bb * m + row][bb * _si + col];

                    }
                for (int row =0; row < bb; row++) {
                    for (int col = 0; col < bb; col++) {
                        short Cvalue = -1;

                        for (int e = 0; e < bb; e++){
                            Cvalue = max(A_elements[id][row][e] + B_elements[id][e][col],   Cvalue);
                        }


                        C[id][row][col] =  max(C[id][row][col],Cvalue);

                    }
                }
            }

            for (int c2 = max(1, bb * c0 - bb-1);
                 c2 <= min(bb * c0 + bb-1, N + bb * c0 - bb * c1 - 1); c2 += 1) { // serial loop
                if (c0 >= 1) {

                    for (int c3 = max(bb * c1, -bb * c0 + bb * c1 + c2); c3 <= min(min(N - 1, bb * c1 + bb-1),
                                                                                   -bb * c0 + bb * c1 + c2 +
                                                                                   bb-1); c3 += 1) {   // parallel loop threads

                        short z = S[-c2 + c3][c3];
                        short _j = (-c2+c3) % bb;
                        short _i = c3 % bb;

                       for (int c4 = 0; c4 < bb-1; c4 += 1)  // blocks 0 (triangles)   // bb-1
                            z = max(S[-c2 + c3][-c2 + c3 + c4 ] + S[-c2 + c3 + c4 + 1][c3], z);

                       z = max(C[id][_j][_i], z);

                        short fragment = (c1 == N/bb-1);

                       for (short c4 =  c2 - bb - fragment; c4 < c2; c4 += 1)   // current tile
                           z = max(S[-c2 + c3][-c2 + c3 + c4] + S[-c2 + c3 + c4 + 1][c3], z);

                        S[-c2 + c3][c3] = max(z, S[-c2 + c3 + 1][c3 - 1] +  paired(seqq[-c2 + c3] , seqq[c3] ));
                    }
                } else {

                    for (int c3 = bb * c1 + c2; c3 <= min(N - 1, bb * c1 + bb-1); c3 += 1) {   // parallel loop threads
                        for (int c4 = 0; c4 < c2; c4 += 1) {  // serial
                            S[-c2 + c3][c3] = max(S[-c2 + c3][-c2 + c3 + c4] + S[-c2 + c3 + c4 + 1][c3],
                                                  S[-c2 + c3][c3]);
                        }
                        S[-c2 + c3][c3] = max(S[-c2 + c3][c3],
                                              S[-c2 + c3 + 1][c3 - 1] + paired(seqq[-c2 + c3], seqq[c3]));
                    }
                }
            }
        }






    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Time taken: %f seconds\n", elapsed_time);

    printf("gpu ended\n");



    for (i = N-1; i >= 0; i--) {
        for (j = i+1; j < N; j++) {
            for (k = 0; k < j-i; k++) {
                S_CPU[i][j] = max(S_CPU[i][k+i] + S_CPU[k+i+1][j], S_CPU[i][j]);
            }

            S_CPU[i][j] = max(S_CPU[i][j], S_CPU[i+1][j-1] + paired(seqq[i],seqq[j]));

        }
    }

    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            if(S[i][j] != S_CPU[i][j]){
                cout << i <<" " <<  j << ":" << S[i][j] << " " << S_CPU[i][j] << endl;
                cout << "error" << endl;
              // exit(1);

            }
   if(1==0)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(S[i][j] < 0)
                cout << "";
            else
                cout << S[i][j];
            cout << "\t";
        }
        cout << "\n";
    }
    cout << endl;
if(1==0)
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(S[i][j] < 0)
                cout << "";
            else
                cout << S_CPU[i][j];
            cout << "\t";
        }
        cout << "\n";
    }
    cout << endl;
    delete[] S;
    delete[] S_CPU;


  return 0;
 }
