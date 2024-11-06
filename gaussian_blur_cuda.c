#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define NUM_THREADS 32      
#define NUM_TEAMS 120

//Giorgos Senteles 3125
#pragma pack(push, 2)
typedef struct bmpheader_ 
{
    char sign[2];
    int size;
    int notused;
    int data;
    int headwidth;
    int width;
    int height;
    short numofplanes;
    short bitpix;
    int method;
    int arraywidth;
    int horizresol;
    int vertresol;
    int colnum;
    int basecolnum;
} bmpheader_t;
#pragma pack(pop)

/* This is the image structure, containing all the BMP information 
 * plus the RGB channels.
 */
typedef struct img_
{
    bmpheader_t header;
    int rgb_width;
    unsigned char *imgdata;
    unsigned char *red;
    unsigned char *green;
    unsigned char *blue;
} img_t;



void gaussian_blur_serial(int, img_t *, img_t *);
void gaussian_blur_omp_device(int radius, img_t *imgin, img_t *imgout);

/* START of BMP utility functions */
static void bmp_read_img_from_file(char *inputfile, img_t *img) 
{
    FILE *file;
    bmpheader_t *header = &(img->header);

    file = fopen(inputfile, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "File %s not found; exiting.\n", inputfile);
        exit(1);
    }
    
    fread(header, sizeof(bmpheader_t), 1, file);
    if (header->bitpix != 24)
    {
        fprintf(stderr, "File %s is not in 24-bit format; exiting.\n", inputfile);
        fclose(file); // Close file before exit
        exit(1);
    }

    img->imgdata = (unsigned char*) calloc(header->arraywidth, sizeof(unsigned char));
    if (img->imgdata == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for image data; exiting.\n");
        fclose(file); // Close file before exit
        exit(1);
    }
    
    fseek(file, header->data, SEEK_SET);
    fread(img->imgdata, header->arraywidth, 1, file);
    fclose(file);
}

static void bmp_clone_empty_img(img_t *imgin, img_t *imgout)
{
    imgout->header = imgin->header;
    imgout->imgdata = (unsigned char*) calloc(imgout->header.arraywidth, sizeof(unsigned char));
    if (imgout->imgdata == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for clone image data; exiting.\n");
        exit(1);
    }
}

static void bmp_write_data_to_file(char *fname, img_t *img) 
{
    FILE *file;
    bmpheader_t *bmph = &(img->header);

    file = fopen(fname, "wb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s for writing; exiting.\n", fname);
        exit(1);
    }
    fwrite(bmph, sizeof(bmpheader_t), 1, file);
    fseek(file, bmph->data, SEEK_SET);
    fwrite(img->imgdata, bmph->arraywidth, 1, file);
    fclose(file);
}

static void bmp_rgb_from_data(img_t *img)
{
    bmpheader_t *bmph = &(img->header);

    int i, j, pos = 0;
    int width = bmph->width, height = bmph->height;
    int rgb_width = img->rgb_width;

    for (i = 0; i < height; i++) 
        for (j = 0; j < width * 3; j += 3, pos++)
        {
            img->red[pos]   = img->imgdata[i * rgb_width + j];
            img->green[pos] = img->imgdata[i * rgb_width + j + 1];
            img->blue[pos]  = img->imgdata[i * rgb_width + j + 2];  
        }
}

static void bmp_data_from_rgb(img_t *img)
{
    bmpheader_t *bmph = &(img->header);
    int i, j, pos = 0;
    int width = bmph->width, height = bmph->height;
    int rgb_width = img->rgb_width;

    for (i = 0; i < height; i++) 
        for (j = 0; j < width * 3; j += 3, pos++) 
        {
            img->imgdata[i * rgb_width + j]     = img->red[pos];
            img->imgdata[i * rgb_width + j + 1] = img->green[pos];
            img->imgdata[i * rgb_width + j + 2] = img->blue[pos];
        }
}

static void bmp_rgb_alloc(img_t *img)
{
    int width, height;

    width = img->header.width;
    height = img->header.height;

    img->red = (unsigned char*) calloc(width * height, sizeof(unsigned char));
    if (img->red == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the red channel; exiting.\n");
        exit(1);
    }

    img->green = (unsigned char*) calloc(width * height, sizeof(unsigned char));
    if (img->green == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the green channel; exiting.\n");
        free(img->red); // Free previously allocated memory
        exit(1);
    }

    img->blue = (unsigned char*) calloc(width * height, sizeof(unsigned char));
    if (img->blue == NULL)
    {
        fprintf(stderr, "Cannot allocate memory for the blue channel; exiting.\n");
        free(img->red);   // Free previously allocated memory
        free(img->green); // Free previously allocated memory
        exit(1);
    }

    img->rgb_width = width * 3;
    if ((width * 3 % 4) != 0) {
        img->rgb_width += (4 - (width * 3 % 4));  
    }
}

static void bmp_img_free(img_t *img)
{
    if (img->red) free(img->red);
    if (img->green) free(img->green);
    if (img->blue) free(img->blue);
    if (img->imgdata) free(img->imgdata);
}

/* END of BMP utility functions */

/* check bounds */
int clamp(int i, int min, int max)
{
    if (i < min) return min;
    else if (i > max) return max;
    return i;  
}

/* Sequential Gaussian Blur */
void gaussian_blur_serial(int radius, img_t *imgin, img_t *imgout)
{
    int i, j;
    int width = imgin->header.width, height = imgin->header.height;
    double row, col;
    double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++) 
        {
            for (row = i - radius; row <= i + radius; row++)
            {
                for (col = j - radius; col <= j + radius; col++) 
                {
                    int x = clamp(col, 0, width - 1);
                    int y = clamp(row, 0, height - 1);
                    int tempPos = y * width + x;
                    double square = (col - j) * (col - j) + (row - i) * (row - i);
                    double sigma = radius * radius;
                    double weight = exp(-square / (2 * sigma)) / (3.14 * 2 * sigma);

                    redSum += imgin->red[tempPos] * weight;
                    greenSum += imgin->green[tempPos] * weight;
                    blueSum += imgin->blue[tempPos] * weight;
                    weightSum += weight;
                }    
            }
            imgout->red[i * width + j] = round(redSum / weightSum);
            imgout->green[i * width + j] = round(greenSum / weightSum);
            imgout->blue[i * width + j] = round(blueSum / weightSum);

            redSum = 0;
            greenSum = 0;
            blueSum = 0;
            weightSum = 0;
        }
    }
}

void gaussian_blur_omp_device(int radius, img_t *imgin, img_t *imgout)
{
    int width = imgin->header.width;
    int height = imgin->header.height;
    unsigned char *in_red = imgin->red;
    unsigned char *in_green = imgin->green;
    unsigned char *in_blue = imgin->blue;
    unsigned char *out_red = imgout->red;
    unsigned char *out_green = imgout->green;
    unsigned char *out_blue = imgout->blue;

    #pragma omp target data map(to: in_red[0:width*height], in_green[0:width*height], in_blue[0:width*height]) \
                            map(from: out_red[0:width*height], out_green[0:width*height], out_blue[0:width*height])
    {
        #pragma omp target teams distribute parallel for collapse(2)  
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++) 
            {
                double redSum = 0.0, greenSum = 0.0, blueSum = 0.0, weightSum = 0.0;
                
                for (int row = i - radius; row <= i + radius; row++)
                {
                    for (int col = j - radius; col <= j + radius; col++) 
                    {
                        int x = clamp(col, 0, width - 1);
                        int y = clamp(row, 0, height - 1);
                        int tempPos = y * width + x;
                        double square = (col - j) * (col - j) + (row - i) * (row - i);
                        double sigma = radius * radius;
                        double weight = exp(-square / (2 * sigma)) / (3.14 * 2 * sigma);

                        redSum += in_red[tempPos] * weight;
                        greenSum += in_green[tempPos] * weight;
                        blueSum += in_blue[tempPos] * weight;
                        weightSum += weight;
                    }    
                }
                out_red[i * width + j] = round(redSum / weightSum);
                out_green[i * width + j] = round(greenSum / weightSum);
                out_blue[i * width + j] = round(blueSum / weightSum);
            }
        }
    }
}

double timeit(void (*func)(int, img_t *, img_t *), int radius, img_t *imgin, img_t *imgout)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    func(radius, imgin, imgout);
    gettimeofday(&end, NULL);
    return (double)(end.tv_usec - start.tv_usec) / 1000000 + (double)(end.tv_sec - start.tv_sec);
}

char *remove_ext(char *str, char extsep, char pathsep) 
{
    char *newstr, *ext, *lpath;

    if (str == NULL) return NULL;
    if ((newstr = malloc(strlen(str) + 1)) == NULL) return NULL;

    strcpy(newstr, str);
    ext = strrchr(newstr, extsep);
    lpath = (pathsep == 0) ? NULL : strrchr(newstr, pathsep);
    if (ext != NULL) 
    {
        if (lpath != NULL) 
        {
            if (lpath < ext) 
                *ext = '\0';
        } 
        else 
            *ext = '\0';
    }
    return newstr;
}

int main(int argc, char *argv[]) 
{
    int radius;
    double exectime_serial = 0.0 , exectime_omp_device= 0.0;
    char *inputfile, *noextfname;   
    char seqoutfile[128], ompoutfile[128];
    img_t imgin, imgout_serial, imgout_omp;

    if (argc < 3)
    {
        fprintf(stderr, "Syntax: %s <blur-radius> <filename>, \n\te.g. %s 2 500.bmp\n", 
                argv[0], argv[0]);
        fprintf(stderr, "Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
        exit(1);
    }

    inputfile = argv[2];
    radius = atoi(argv[1]);
    if (radius < 0)
    {
        fprintf(stderr, "Radius should be an integer >= 0; exiting.\n");
        exit(1);
    }

    noextfname = remove_ext(inputfile, '.', '/');
    sprintf(seqoutfile, "%s-r%d-serial.bmp", noextfname, radius);
    sprintf(ompoutfile, "%s-r%d-omp.bmp", noextfname, radius);

    bmp_read_img_from_file(inputfile, &imgin);

    bmp_clone_empty_img(&imgin, &imgout_serial);
    bmp_rgb_alloc(&imgin);
    bmp_rgb_alloc(&imgout_serial);

    bmp_clone_empty_img(&imgin, &imgout_omp);
    bmp_rgb_alloc(&imgout_omp);

    printf("<<< Gaussian Blur (h=%d,w=%d,r=%d) >>>\n", imgin.header.height, 
           imgin.header.width, radius);

    /* Image data to R,G,B */
    bmp_rgb_from_data(&imgin);

    /* Run & time serial Gaussian Blur */
    exectime_serial = timeit(gaussian_blur_serial, radius, &imgin, &imgout_serial);


    exectime_omp_device = timeit(gaussian_blur_omp_device, radius, &imgin, &imgout_omp);

    /* Save the results (serial) */
    bmp_data_from_rgb(&imgout_serial);
    bmp_write_data_to_file(seqoutfile, &imgout_serial);

    /* Save the results (OpenMP device) */
    bmp_data_from_rgb(&imgout_omp);
    bmp_write_data_to_file(ompoutfile, &imgout_omp);

    printf("Total execution time (sequential): %lf\n", exectime_serial);
    printf("Total execution time (omp_device): %lf\n", exectime_omp_device);

    bmp_img_free(&imgin);
    bmp_img_free(&imgout_serial);
    bmp_img_free(&imgout_omp);

    free(noextfname);

    return 0;
}
