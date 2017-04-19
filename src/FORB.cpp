/**
 * File: FORB.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 * Distance function has been modified 
 *
 */
 
#if __SSE2__
#include <emmintrin.h>
#endif
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <limits.h>

#include <glog/logging.h>

#include "DUtils/DUtils.h"
#include "DVision/DVision.h"
#include "DBoW2/FORB.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

const int FORB::L=32;

void FORB::meanValue(const std::vector<FORB::pDescriptor> &descriptors, 
  FORB::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    mean.release();
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    vector<int> sum(FORB::L * 8, 0);
    
    for(size_t i = 0; i < descriptors.size(); ++i)
    {
      const cv::Mat &d = *descriptors[i];
      const unsigned char *p = d.ptr<unsigned char>();
      
      for(int j = 0; j < d.cols; ++j, ++p)
      {
        if(*p & (1 << 7)) ++sum[ j*8     ];
        if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
        if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
        if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
        if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
        if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
        if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
        if(*p & (1))      ++sum[ j*8 + 7 ];
      }
    }
    
    mean = cv::Mat::zeros(1, FORB::L, CV_8U);
    unsigned char *p = mean.ptr<unsigned char>();
    
    const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2;
    for(size_t i = 0; i < sum.size(); ++i)
    {
      if(sum[i] >= N2)
      {
        // set bit
        *p |= 1 << (7 - (i % 8));
      }
      
      if(i % 8 == 7) ++p;
    }
  }
}

// --------------------------------------------------------------------------
  
#if __SSE2__

// Courtesy of
// http://stackoverflow.com/questions/17354971/fast-counting-the-number-of-set-bits-in-m128i-register
namespace FORB_internal {

const __m128i popcount_mask1 = _mm_set1_epi8(0x77);
const __m128i popcount_mask2 = _mm_set1_epi8(0x0F);
inline __m128i popcnt8(__m128i x) {
    __m128i n;
    // Count bits in each 4-bit field.
    n = _mm_srli_epi64(x, 1);
    n = _mm_and_si128(popcount_mask1, n);
    x = _mm_sub_epi8(x, n);
    n = _mm_srli_epi64(n, 1);
    n = _mm_and_si128(popcount_mask1, n);
    x = _mm_sub_epi8(x, n);
    n = _mm_srli_epi64(n, 1);
    n = _mm_and_si128(popcount_mask1, n);
    x = _mm_sub_epi8(x, n);
    x = _mm_add_epi8(x, _mm_srli_epi16(x, 4));
    x = _mm_and_si128(popcount_mask2, x);
    return x;
}

inline __m128i popcnt64(__m128i n) {
    const __m128i cnt8 = popcnt8(n);
    return _mm_sad_epu8(cnt8, _mm_setzero_si128());
}

inline int popcnt128(__m128i n) {
    const __m128i cnt64 = popcnt64(n);
    const __m128i cnt64_hi = _mm_unpackhi_epi64(cnt64, cnt64);
    const __m128i cnt128 = _mm_add_epi32(cnt64, cnt64_hi);
    return _mm_cvtsi128_si32(cnt128);
}

}  // namespace FORB_internal

#endif  // __SSE2__

int FORB::distance(const FORB::TDescriptor &a,
  const FORB::TDescriptor &b)
{
  int dist=0;

#if __SSE2__
  for (int i = 0; i < 8; i += 4)
  {
    const __m128i* ai = (const __m128i*)(a.ptr<int32_t>() + i);
    const __m128i* bi = (const __m128i*)(b.ptr<int32_t>() + i);

    __m128i v = _mm_xor_si128(*ai, *bi);
    dist += FORB_internal::popcnt128(v);
  }
#else
  // Bit set count operation from
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  for(int i=0; i<8; i++, pa++, pb++)
  {
      unsigned  int v = *pa ^ *pb;
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }
#endif

  return dist;
}

// --------------------------------------------------------------------------
  
std::string FORB::toString(const FORB::TDescriptor &a)
{
  stringstream ss;
  const unsigned char *p = a.ptr<unsigned char>();
  
  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << (int)*p << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FORB::fromString(FORB::TDescriptor &a, const std::string &s)
{
  a.create(1, FORB::L, CV_8U);
  unsigned char *p = a.ptr<unsigned char>();
  
  stringstream ss(s);
  for(int i = 0; i < FORB::L; ++i, ++p)
  {
    int n;
    ss >> n;
    
    if(!ss.fail()) 
      *p = (unsigned char)n;
  }
  
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const size_t N = descriptors.size();
  
  mat.create(N, FORB::L*8, CV_32F);
  float *p = mat.ptr<float>();
  
  for(size_t i = 0; i < N; ++i)
  {
    const int C = descriptors[i].cols;
    const unsigned char *desc = descriptors[i].ptr<unsigned char>();
    
    for(int j = 0; j < C; ++j, p += 8)
    {
      p[0] = (desc[j] & (1 << 7) ? 1 : 0);
      p[1] = (desc[j] & (1 << 6) ? 1 : 0);
      p[2] = (desc[j] & (1 << 5) ? 1 : 0);
      p[3] = (desc[j] & (1 << 4) ? 1 : 0);
      p[4] = (desc[j] & (1 << 3) ? 1 : 0);
      p[5] = (desc[j] & (1 << 2) ? 1 : 0);
      p[6] = (desc[j] & (1 << 1) ? 1 : 0);
      p[7] = desc[j] & (1);
    }
  } 
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{

  descriptors.convertTo(mat, CV_32F);
  return; 

  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.rows;
  const int C = descriptors.cols;
  
  mat.create(N, FORB::L*8, CV_32F);
  float *p = mat.ptr<float>(); // p[i] == 1 or 0
  
  const unsigned char *desc = descriptors.ptr<unsigned char>();
  
  for(int i = 0; i < N; ++i, desc += C)
  {
    for(int j = 0; j < C; ++j, p += 8)
    {
      p[0] = (desc[j] & (1 << 7) ? 1 : 0);
      p[1] = (desc[j] & (1 << 6) ? 1 : 0);
      p[2] = (desc[j] & (1 << 5) ? 1 : 0);
      p[3] = (desc[j] & (1 << 4) ? 1 : 0);
      p[4] = (desc[j] & (1 << 3) ? 1 : 0);
      p[5] = (desc[j] & (1 << 2) ? 1 : 0);
      p[6] = (desc[j] & (1 << 1) ? 1 : 0);
      p[7] = desc[j] & (1);
    }
  } 
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  mat.create(descriptors.size(), FORB::L, CV_8U);
  
  unsigned char *p = mat.ptr<unsigned char>();
  
  for(size_t i = 0; i < descriptors.size(); ++i, p += FORB::L)
  {
    const unsigned char *d = descriptors[i].ptr<unsigned char>();
    std::copy(d, d + FORB::L, p);
  }
  
}

// --------------------------------------------------------------------------

} // namespace DBoW2

