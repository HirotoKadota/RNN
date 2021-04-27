#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define _USE_MATH_DEFINES
#define HID_LAYER_NUM 10
#define TIME_LEN 20
#define DATA_SIZE 2
#define LR 0.001
#define EPOCH 20000
#define TIME_STEP 200
#define SD 1e-2

int N = (int) ( EPOCH / TIME_STEP);
int getrand();
double rand_normal(double mu, double sigma);
double mean_squared_error(const double [][TIME_LEN]);
void draw_graph(FILE *fp, double y[][TIME_LEN]);
void loss_plot(FILE *fp, double loss[TIME_STEP]);

int main(){
	double x[DATA_SIZE][TIME_LEN];
	double y[DATA_SIZE][TIME_LEN];
	double x_[DATA_SIZE][TIME_LEN];
    double w_h[HID_LAYER_NUM][TIME_LEN];
	double w_1[HID_LAYER_NUM][DATA_SIZE];
	double w_2[DATA_SIZE][HID_LAYER_NUM];
	double b_1[HID_LAYER_NUM][TIME_LEN];
	double b_2[DATA_SIZE][TIME_LEN];
	double y_pred[DATA_SIZE][TIME_LEN];
	double mse[DATA_SIZE][TIME_LEN];

	int i, j;

	for(i = 0; i < TIME_LEN; i++){
		x[0][i] = 0.8 * sin(2 * M_PI * i / TIME_LEN) + rand_normal(0.0, SD);
		x[1][i] = 0.8 * sin(2 * M_PI * 2 * i / TIME_LEN) + rand_normal(0.0, SD);
		y[0][i] = 0.8 * sin(2 * M_PI * (i + 1) / TIME_LEN);
		y[1][i] = 0.8 * sin(2 * M_PI * 2 * (i + 1) / TIME_LEN);

		x_[0][i] = 0.8 * sin(2 * M_PI * i / TIME_LEN) + rand_normal(0.0, SD);
		x_[1][i] = 0.8 * sin(2 * M_PI * 2 * i / TIME_LEN) + rand_normal(0.0, SD);
	}

	for(i = 0; i < HID_LAYER_NUM; i++){
		w_1[i][0] = getrand();
		w_1[i][1] = getrand();
		w_2[0][i] = getrand();
		w_2[1][i] = getrand();
	}

	for(i = 0; i < HID_LAYER_NUM; i++){
        for(j = 0; j < HID_LAYER_NUM; j++){
            w_h[i][j] = getrand();
        }
    }

	for(i = 0; i < HID_LAYER_NUM; i++){
		for(j = 0; j < TIME_LEN; j++){
			b_1[i][j] = getrand();
		}
	}

	for(i = 0; i < DATA_SIZE; i++){
		for(j = 0; j < TIME_LEN; j++){
			b_2[i][j] = getrand();
		}
	}

	double hid_out[HID_LAYER_NUM][TIME_LEN]; //hid_out = s_t
    double hid_in[HID_LAYER_NUM][TIME_LEN]; //hid_in = s_t-1
	double Lu[HID_LAYER_NUM][TIME_LEN];
	double Lo[HID_LAYER_NUM][TIME_LEN];
	int k, l;
	double mse_[TIME_STEP];
	int flag = 0;
	int ep = 0;

	while(ep < EPOCH + 1){
		/* initialize hid_in & hid_out & Lo & Lu */
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < HID_LAYER_NUM; j++){
				hid_out[j][i] = 0;
				Lo[j][i] = 0;
				Lu[j][i] = 0;
				if(i == 0 && ep == 0) hid_in[j][0] = getrand();
				else if(i == 0) hid_in[j][0] = hid_in[j][0];
				else hid_in[j][i] = 0;
			}
		}

		/* initialize y_pred */
		for(i = 0; i < DATA_SIZE; i++){
			for(j = 0; j < TIME_LEN; j++){
				y_pred[i][j] = 0;
			}
		}

		/* hid_out = tanh( x * w_1 + w_h * hid_in + b_1 ) */ 
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < HID_LAYER_NUM; j++){
				for(k = 0; k < DATA_SIZE; k++){
					if(ep == EPOCH){
						hid_out[j][i] += w_1[j][k] * x_[k][i];
					}else{
						hid_out[j][i] += w_1[j][k] * x[k][i];
					}
				}
				hid_out[j][i] += b_1[j][i];

				for(k = 0; k < HID_LAYER_NUM; k++){
					for(l = 0; l < HID_LAYER_NUM; l++){
						hid_out[j][i] += w_h[k][l] * hid_in[l][i];
					}
				}

				hid_out[j][i] = tanh(hid_out[j][i]);

				if(i < TIME_LEN - 1) hid_in[j][i + 1] = hid_out[j][i];
			}
		}

		/* y_pred = tanh( hid_out * w_2 + b_2 ) */
		for(i = 0; i < DATA_SIZE; i++){
			for(j = 0; j < TIME_LEN; j++){
				for(k = 0; k < HID_LAYER_NUM; k++){
					y_pred[i][j] += w_2[i][k] * hid_out[k][j];
				}
				y_pred[i][j] += b_2[i][j];
				y_pred[i][j] = tanh(y_pred[i][j]);
			}
		}

		if(ep == EPOCH) break;

		/* change the value of parameters
			w_1 = w_1 - LR * Lu * x
			b_1 = b_1 - LR * Lu
			w_h = w_h - LR * Lu * hid_in
			w_2 = w_2 - LR * Lo * hid_out
			b_2 = b_2 - LR * Lo
			s0 = hid_in[][0] - LR * sum(w_h * Lu[0])
		*/

		//Lo = (y_pred - y) * (1 - y_pred * y_pred)
		for(i = 0; i < DATA_SIZE; i++){
			for(j = 0; j < TIME_LEN; j++){
				Lo[i][j] += (y_pred[i][j] - y[i][j]) * (1 - y_pred[i][j] * y_pred[i][j]);
			}
		}

		//Lu[][t] = dL / du_t = ( sum( w_2 * Lo ) + sum( w_h * Lu[][t + 1]) ) * (1 - hid_out * hid_out)
		for(i = TIME_LEN; i >= 0; i--){
			// + sum( w_2 * Lo )
			for(j = 0; j < HID_LAYER_NUM; j++){
				for(k = 0; k < DATA_SIZE; k++){
					Lu[j][i] += w_2[k][j] * Lo[k][i];
				}
			}

			// + sum( w_h * Lu[][t + 1] )
			if(i < TIME_LEN - 1){
				for(j = 0; j < HID_LAYER_NUM; j++){
					for(k = 0; k < HID_LAYER_NUM; k++){
						Lu[j][i] += w_h[k][j] * Lu[k][i + 1];
					}
				}
			}

			// * ds / du
			for(j = 0; j < HID_LAYER_NUM; j++){
				Lu[j][i] *= (1 - hid_out[j][i] * hid_out[j][i]);
			}
		}

		//s0 = hid_in[][0]
		for(i = 0; i < HID_LAYER_NUM; i++){
			for(j = 0; j < HID_LAYER_NUM; j++){
				hid_in[i][0] -= LR * w_h[i][j] * Lu[i][0];
			}
		}

		//w_1
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < DATA_SIZE; j++){
				for(k = 0; k < HID_LAYER_NUM; k++){
					w_1[k][j] -= LR * Lu[k][i] * x[j][i];
				}
			}
		}

		//b_1
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < HID_LAYER_NUM; j++){
				b_1[j][i] -= LR * Lu[j][i];
			}
		}

		//w_h
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < HID_LAYER_NUM; j++){
				w_h[j][i] -= LR * Lu[j][i] * hid_in[j][i];
			}
		}

		//w_2
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < DATA_SIZE; j++){
				for(k = 0; k < HID_LAYER_NUM; k++){
					w_2[j][k] -= LR * Lo[j][i] * hid_out[k][i];
				}
			}
		}

		//b_2
		for(i = 0; i < TIME_LEN; i++){
			for(j = 0; j < DATA_SIZE; j++){
				b_2[j][i] -= LR * Lo[j][i];
			}
		}

		//0.5 * squared error
		for(i = 0; i < DATA_SIZE; i++){
			for(j = 0; j < TIME_LEN; j++){
				mse[i][j] = 0.5 * (y_pred[i][j] - y[i][j]) * (y_pred[i][j] - y[i][j]);
			}
		}

		if(ep % N == 0){
			mse_[flag] = mean_squared_error(mse);
			printf("epoch : %d\n", ep);
			printf("mean squared error : %.7f\n", mse_[flag]);
			flag ++;
		}

		ep ++;
	}

	printf("sin_pred\tsin2_pred\tsin\tsin2\n");
	for(i = 0; i < TIME_LEN; i++){
		printf("%.3f\t%.3f\t%.3f\t%.3f\n", y_pred[0][i], y_pred[1][i], y[0][i], y[1][i]);
	}

	FILE *fp, *gp, *hp;
	draw_graph(fp, y_pred);
	draw_graph(gp, y);
	loss_plot(hp, mse_);

	return 0;
}

int getrand(){
	srand((unsigned int)time(NULL));
	return rand() / (double)RAND_MAX - 0.5;
}

double Uniform(void){
	return ( (double) rand() + 1.0 ) / ( (double) RAND_MAX + 2.0 );
}

double rand_normal( double mu, double sigma ){
	double z = sqrt( -2.0 * log( Uniform() ) ) * sin( 2.0 * M_PI * Uniform() );
	return mu + sigma * z;
}

double mean_squared_error(const double err[][TIME_LEN]){
	int i, j;
	double sum_err = 0.0;

	for(i = 0; i < 2; i++){
		for(j = 0; j < TIME_LEN; j++){
			sum_err += err[i][j];
		}
	}
	return sum_err / TIME_LEN;
}

void draw_graph(FILE *fp, double y[][TIME_LEN]){
	fp = popen("/usr/local/bin/gnuplot -persist", "w");
	fprintf(fp, "unset key\n");

	if( y[1][0] - 0.8 * sin(2 * 2 * M_PI / TIME_LEN) != 0.0 ){
		fprintf(fp, "set title 'Predicted Lissajous Curve with Noise'\n");
	}else{
		fprintf(fp, "set title 'Lissajous Curve'\n");
	}
	fprintf(fp, "set title font 'Times New Roman,25'\n");

	fprintf(fp, "set tics font 'Times New Roman,17'\n");
	fprintf(fp, "set xrange [-1.5:1.5]\n");
	fprintf(fp, "set yrange [-1.5:1.5]\n");
	fprintf(fp, "plot '-' with points pt 6 ps 2\n");

	for(int i = 0; i < TIME_LEN; i++){
		fprintf(fp, "%f\t%f\n", y[0][i], y[1][i]);
	}
	fprintf(fp, "e\n");

	pclose(fp);
}

void loss_plot(FILE *fp, double loss[TIME_STEP]){
	int i;
	int x[TIME_STEP];

	for(i = 0; i < TIME_STEP; i++){
		x[i] = N * i;
	}

	fp = popen("/usr/local/bin/gnuplot -persist", "w");
	fprintf(fp, "unset key\n");
	fprintf(fp, "set title 'Loss'\n");
	fprintf(fp, "set title font 'Times New Roman,25'\n");

	fprintf(fp, "set tics font 'Times New Roman,17'\n");
	fprintf(fp, "set xrange [0:EPOCH]\n");
	fprintf(fp, "set yrange [0:0.6]\n");
	fprintf(fp, "plot '-' with lines linetype 1\n");

	for(i = 0; i < TIME_STEP; i++){
		fprintf(fp, "%d\t%f\n", x[i], loss[i]);
	}
	fprintf(fp, "e\n");

	pclose(fp);
}