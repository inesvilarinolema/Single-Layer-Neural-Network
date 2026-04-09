#include <stdio.h>
#include <math.h>

/*Ines Vilariño Lema*/

//Weights for Neurons 
float w1[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};
float w2[13]={0,0,0,0,0,0,0,0,0,0,0,0,0};

//Patterns: 1,4,7,2
float x[4][12] = {{1,1,-1,1,1,-1,1,1,-1,1,1,-1}, {-1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1}, {-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1}, {-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1}};

//Desired outputs
float d[4][2] = {{0,0},{1,0},{0,1}, {1,1}};

float s1= 0.0;
float s2=0.0;
float y2=0.0;
float yy1 = 0.0;

float error1[4] = {0,0,0,0};
float error2[4] = {0,0,0,0};

float alfa = 0.8;
float beta = 0.6;
float eMax = 0.0001;

int iteraciones = -1;
float total_error=0;

//unipolar sigmoid
float fx (float x){
    float pot = exp(-beta * x);
    float result = 1.0 / (1 + pot);

    return result;
}

//derivative
float fderivate (float x){
    float first = beta * fx(x);
    float sec = (1 - fx(x));

    return first * sec;
}

int main(){

    while(1){
        total_error = 0;

        for(int i=0; i<4; i++){
            s1 += w1[0];
            s2 += w2[0];

            for(int m=1; m<13; m++){
                s1+= w1[m]*x[i][m-1];
                s2+= w2[m]*x[i][m-1];
            }

            yy1 = fx(s1);
            y2 = fx(s2);

            //Neuron 1 Update
            float delta1 = (d[i][0] - yy1) * fderivate(s1);
            w1[0] = w1[0] + (alfa * delta1); //Bias update
            for(int j=1; j<13; j++){
                w1[j] = w1[j] + (alfa * delta1 * x[i][j-1]);
            }

            //Neuron 2 Update
            float delta2 = (d[i][1] - y2) * fderivate(s2);
            w2[0] = w2[0] + (alfa * delta2); //Bias update
            for(int j=1; j<13; j++){
                w2[j] = w2[j] + (alfa * delta2 * x[i][j-1]);
            }

            total_error += 0.5 * pow(d[i][0] - yy1, 2);
            total_error += 0.5 * pow(d[i][1] - y2, 2);
        }
        
        iteraciones++;

        if(iteraciones % 109 == 0) {
            printf("Interaction: %d, Total error:: %f\n", iteraciones, total_error);
        }

        if(total_error <= eMax){
            printf("\n--- CONVERGENCE REACHED ---\n");
            printf("\n Final error: %f, Final error: %f\n", total_error);
            break;
        }
    }

    printf("\n\n---CHECKING RESULTS---\n");

    int digits[4] = {1,4,7,2};
    for(int i=0; i<4; i++){
        s1 += w1[0];
        s2 += w2[0];

        for(int m=1; m<13; m++){
            s1+= w1[m]*x[i][m-1];
            s2+= w2[m]*x[i][m-1];
        }

        yy1 = fx(s1);
        y2 = fx(s2);


        printf("Digit %d | Target: %.0f %.0f | Output: %.4f %.4f\n", 
               digits[i], d[i][0], d[i][1], yy1, y2);

    }

    float xnew1[12] = {1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1};
    float xnew2[12] = {1, 1, -1, 1, -1, -1, 1, 1,- 1, 1, 1, -1};

    printf("\n---TESTING NEW INPUTS---\n");
    
    s1 += w1[0];
    s2 += w2[0];

    for(int m=1; m<13; m++){
        s1+= w1[m]*xnew1[m-1];
        s2+= w2[m]*xnew1[m-1];
    }

    printf("New Input 1 -> N1: %.4f, N2: %.4f\n", fx(s1), fx(s2));

    s1=0;
    s2=0;
    s1 += w1[0];
    s2 += w2[0];

    for(int m=1; m<13; m++){
        s1+= w1[m]*xnew2[m-1];
        s2+= w2[m]*xnew2[m-1];
    }

    printf("New Input 2 -> N1: %.4f, N2: %.4f\n", fx(s1), fx(s2));

    return 0;   
}