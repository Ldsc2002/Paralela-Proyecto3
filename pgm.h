#ifndef PGM_H
#define PGM_H

class PGMImage
{
 private:
   int x_dim;
   int y_dim;
   int num_colors;
   unsigned char *pixels;
   std::vector<int> color;
 
 public:
   PGMImage(char *fname, int programID);
   PGMImage(int x, int y, int col);
   ~PGMImage(void);
   void write(const char* filename, std::vector<std::pair<int, int>> lines, float radInc, int rBins);
   int getXDim(void);
   int getYDim(void);
   int getNumColors(void);
   unsigned char* getPixels(void);
};

#endif
