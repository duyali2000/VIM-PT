#include <stdio.h>
void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % dn " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNGE ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr9 , 0 , n9 - 1 ) ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { printf ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = INT_MIN ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MIN ) printf ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element STRNEWLINE " ) ; else printf ( " The ▁ second ▁ largest ▁ element ▁ is ▁ % dn " , second ) ; }
int main ( ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2largest ( arr , n ) ; return 0 ; }
#include <stdio.h>
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { int arr [ ] = { 5 , 5 , 10 , 100 , 10 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ n " , FindMaxSum ( arr , n ) ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
int minJumps ( int arr [ ] , int l , int h ) {
if ( h == l ) return 0 ;
int min = INT_MAX ; for ( int i = l + 1 ; i <= h && i <= l + arr [ l ] ; i ++ ) { int jumps = minJumps ( arr , i , h ) ; if ( jumps != INT_MAX && jumps + 1 < min ) min = jumps + 1 ; } return min ; }
int main ( ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ end ▁ is ▁ % d ▁ " , minJumps ( arr , 0 , n - 1 ) ) ; return 0 ; }
#include <stdio.h>
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ % d STRNEWLINE " , maxSumIS ( arr , n ) ) ; return 0 ; }
#include <stdio.h>
void moveToEnd ( int mPlusN [ ] , int size ) { int i = 0 , j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) if ( mPlusN [ i ] != NA ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } }
int merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( j == n ) || ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) {
int mPlusN [ ] = { 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = sizeof ( N ) / sizeof ( N [ 0 ] ) ; int m = sizeof ( mPlusN ) / sizeof ( mPlusN [ 0 ] ) - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { printf ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } printf ( " ▁ The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ % d ▁ and ▁ % d " , arr [ min_l ] , arr [ min_r ] ) ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : swap ( & a [ lo ++ ] , & a [ mid ++ ] ) ; break ; case 1 : mid ++ ; break ; case 2 : swap ( & a [ mid ] , & a [ hi -- ] ) ; break ; } } }
void printArray ( int arr [ ] , int arr_size ) { int i ; for ( i = 0 ; i < arr_size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " n " ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; sort012 ( arr , arr_size ) ; printf ( " array ▁ after ▁ segregation ▁ " ) ; printArray ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
int findNumberOfTriangles ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( arr [ 0 ] ) , comp ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
int main ( ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Total ▁ number ▁ of ▁ triangles ▁ possible ▁ is ▁ % d ▁ " , findNumberOfTriangles ( arr , size ) ) ; return 0 ; }
#include <stdio.h>
int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ;
if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; key = 10 ; printf ( " Index : ▁ % d STRNEWLINE " , binarySearch ( arr , 0 , n - 1 , key ) ) ; return 0 ; }
#include <stdio.h>
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d " , equilibrium ( arr , arr_size ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return -1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 20 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) printf ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " , x ) ; else printf ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " , x , arr [ index ] ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int NEW_LINE int findCandidate ( int * , int ) ; bool isMajority ( int * , int , int ) ;
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
bool isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) if ( a [ i ] == cand ) count ++ ; if ( count > size / 2 ) return 1 ; else return 0 ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) printf ( " ▁ % d ▁ " , cand ) ; else printf ( " No ▁ Majority ▁ Element " ) ; }
int main ( ) { int a [ ] = { 1 , 3 , 3 , 1 , 2 } ; int size = ( sizeof ( a ) ) / sizeof ( a [ 0 ] ) ;
printMajority ( a , size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
void printRepeating ( int arr [ ] , int size ) { int i , j ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) for ( j = i + 1 ; j < size ; j ++ ) if ( arr [ i ] == arr [ j ] ) printf ( " ▁ % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #include <math.h>
int fact ( int n ) ; void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; printf ( " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d " , x , y ) ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) {
int xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) xor ^= i ;
set_bit_no = xor & ~ ( xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} printf ( " n ▁ The ▁ two ▁ repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d ▁ " , x , y ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
void printRepeating ( int arr [ ] , int size ) { int i ; printf ( " The repeating elements are " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) ] > 0 ) arr [ abs ( arr [ i ] ) ] = - arr [ abs ( arr [ i ] ) ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return -1 ; }
int main ( ) { int arr [ 10 ] = { -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Fixed ▁ Point ▁ is ▁ % d " , binarySearch ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
bool find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { printf ( " Triplet ▁ is ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] ) ; return true ; } } } }
return false ; }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; find3Numbers ( A , arr_size , sum ) ; return 0 ; }
#include <stdio.h> NEW_LINE int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int result = search ( arr , n , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
#include <stdio.h>
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE #define RANGE  255
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; printf ( " Sorted ▁ character ▁ array ▁ is ▁ % sn " , arr ) ; return 0 ; }
#include <stdio.h>
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; printf ( " Following ▁ activities ▁ are ▁ selected ▁ n " ) ;
i = 0 ; printf ( " % d ▁ " , i ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { printf ( " % d ▁ " , j ) ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ;
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int minCost ( int cost [ R ] [ C ] , int m , int n ) { if ( n < 0 m < 0 ) return INT_MAX ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ; int minCost ( int cost [ R ] [ C ] , int m , int n ) { int i , j ;
int tc [ R ] [ C ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h>
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
#include <stdio.h>
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h>
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdbool.h> NEW_LINE #include <stdio.h>
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
bool findPartiion ( int arr [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
int main ( ) { int arr [ ] = { 3 , 1 , 5 , 9 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) printf ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ " " of ▁ equal ▁ sum " ) ; else printf ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets " " ▁ of ▁ equal ▁ sum " ) ; return 0 ; }
#include <stdio.h>
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) printf ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ % d ▁ keystrokes ▁ is ▁ % d STRNEWLINE " , N , findoptimal ( N ) ) ; }
#include <stdio.h> NEW_LINE #include <string.h>
void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
#include <stdio.h> NEW_LINE float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; printf ( " % f " , power ( x , y ) ) ; return 0 ; }
#include <stdio.h>
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = -1 , m2 = -1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) { m1 = m2 ;
m2 = ar1 [ i ] ; i ++ ; } else { m1 = m2 ;
m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
int main ( ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) printf ( " Median ▁ is ▁ % d " , getMedian ( ar1 , ar2 , n1 ) ) ; else printf ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ of ▁ unequal ▁ size " ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
int main ( ) { printf ( " % d " , multiply ( 5 , -11 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
int main ( ) { printf ( " % d " , pow ( 5 , 3 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h>
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
int findSmallerInRight ( char * str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; int i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
int main ( ) { char str [ ] = " string " ; printf ( " % d " , findRank ( str ) ) ; return 0 ; }
#include <stdio.h>
int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int main ( ) { int n = 8 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
printf ( " % d ▁ " , C ) ; C = C * ( line - i ) / i ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
#include <stdio.h>
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; printf ( " e ^ x ▁ = ▁ % f " , exponential ( n , x ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) ;
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) printf ( " % d ▁ " , data [ j ] ) ; printf ( " STRNEWLINE " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) printf ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
int hour_angle = 0.5 * ( h * 60 + m ) ; int minute_angle = 6 * m ;
int angle = abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
int main ( ) { printf ( " % d ▁ n " , calcAngle ( 9 , 60 ) ) ; printf ( " % d ▁ n " , calcAngle ( 3 , 30 ) ) ; return 0 ; }
#include <stdio.h>
int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define INT_SIZE  32 NEW_LINE int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
int main ( ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int smallest ( int x , int y , int z ) { int c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int addOne ( int x ) { return ( - ( ~ x ) ) ; }
int main ( ) { printf ( " % d " , addOne ( 13 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int
bool isPowerOfFour ( unsigned int n ) { int count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
#include <stdio.h>
int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
#include <stdio.h>
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; printf ( " % d " , countSetBits ( i ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; printf ( " % d STRNEWLINE " , countSetBitsRec ( num ) ) ; }
#include <stdio.h> NEW_LINE unsigned int nextPowerOf2 ( unsigned int n ) { unsigned int p = 1 ; if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( p < n ) p <<= 1 ; return p ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
#include <stdio.h>
unsigned int nextPowerOf2 ( unsigned int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # define bool  int
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; printf ( " Parity ▁ of ▁ no ▁ % d ▁ = ▁ % s " , n , ( getParity ( n ) ? " odd " : " even " ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h> NEW_LINE #include <math.h>
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h>
unsigned int swapBits ( unsigned int x ) {
unsigned int even_bits = x & 0xAAAAAAAA ;
unsigned int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
int main ( ) {
unsigned int x = 23 ;
printf ( " % u ▁ " , swapBits ( x ) ) ; return 0 ; }
#include <stdio.h>
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned i = 1 , pos = 1 ;
while ( ! ( i & n ) ) {
i = i << 1 ;
++ pos ; } return pos ; }
int main ( void ) { int n = 16 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; return 0 ; }
#include <stdio.h>
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; printf ( " Array ▁ after ▁ segregation ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
void nextGreatest ( int arr [ ] , int size ) {
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = -1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; nextGreatest ( arr , size ) ; printf ( " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ) ; printArray ( arr , size ) ; return ( 0 ) ; }
#include <stdio.h>
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; int i , j ; for ( i = 0 ; i < arr_size ; i ++ ) { for ( j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
int main ( ) { int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ;
printf ( " Maximum ▁ difference ▁ is ▁ % d " , maxDiff ( arr , 5 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ maximum ▁ element ▁ is ▁ % d " , findMaximum ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int getMissingNo ( int a [ ] , int n ) { int i , total ; total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int a [ ] = { 1 , 2 , 4 , 5 , 6 } ; int miss = getMissingNo ( a , 5 ) ; printf ( " % d " , miss ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void printTwoElements ( int arr [ ] , int size ) { int i ; printf ( " The repeating element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } printf ( " and the missing element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) printf ( " % d " , i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; return 0 ; }
#include <stdio.h>
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } printf ( " The two ODD elements are % d & % d " }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { printf ( " Pair ▁ Found : ▁ ( % d , ▁ % d ) " , arr [ i ] , arr [ j ] ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } printf ( " No ▁ such ▁ pair " ) ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
#include <stdio.h>
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) printf ( " % d , ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] , A [ l ] ) ; } } } }
int main ( ) { int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int X = 91 ; findFourElements ( A , n , X ) ; return 0 ; }
#include <stdio.h>
const int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
void Kroneckerproduct ( int A [ ] [ cola ] , int B [ ] [ colb ] ) { int C [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; printf ( " % d TABSYMBOL " , C [ i + l + 1 ] [ j + k + 1 ] ) ; } } printf ( " STRNEWLINE " ) ; } } }
int main ( ) { int A [ 3 ] [ 2 ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } , B [ 2 ] [ 3 ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; return 0 ; }
#include <stdio.h> NEW_LINE int Identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) printf ( " % d ▁ " , 1 ) ; else printf ( " % d ▁ " , 0 ) ; } printf ( " STRNEWLINE " ) ; } return 0 ; }
int main ( ) { int size = 5 ; identity ( size ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define N  4
void subtract ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; printf ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) printf ( " % d ▁ " , C [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return 0 ; }
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
#define NA  -1
http : int comp ( const void * a , const void * b ) { return * ( int * ) a > * ( int * ) b ; }
int min ( int x , int y ) { return ( x < y ) ? x : y ; }
#include <stdio.h>
void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % dn " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNGE ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr9 , 0 , n9 - 1 ) ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { printf ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = INT_MIN ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MIN ) printf ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element STRNEWLINE " ) ; else printf ( " The ▁ second ▁ largest ▁ element ▁ is ▁ % dn " , second ) ; }
int main ( ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2largest ( arr , n ) ; return 0 ; }
#include <stdio.h>
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { int arr [ ] = { 5 , 5 , 10 , 100 , 10 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ n " , FindMaxSum ( arr , n ) ) ; return 0 ; }
#include <limits.h> NEW_LINE #include <stdio.h>
int minJumps ( int arr [ ] , int l , int h ) {
if ( h == l ) return 0 ;
if ( arr [ l ] == 0 ) return INT_MAX ;
int min = INT_MAX ; for ( int i = l + 1 ; i <= h && i <= l + arr [ l ] ; i ++ ) { int jumps = minJumps ( arr , i , h ) ; if ( jumps != INT_MAX && jumps + 1 < min ) min = jumps + 1 ; } return min ; }
int main ( ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ end ▁ is ▁ % d ▁ " , minJumps ( arr , 0 , n - 1 ) ) ; return 0 ; }
#include <stdio.h>
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ % d STRNEWLINE " , maxSumIS ( arr , n ) ) ; return 0 ; }
#include <stdio.h>
void moveToEnd ( int mPlusN [ ] , int size ) { int i = 0 , j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) if ( mPlusN [ i ] != NA ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } }
int merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( j == n ) || ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) {
int mPlusN [ ] = { 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = sizeof ( N ) / sizeof ( N [ 0 ] ) ; int m = sizeof ( mPlusN ) / sizeof ( mPlusN [ 0 ] ) - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { printf ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } printf ( " ▁ The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ % d ▁ and ▁ % d " , arr [ min_l ] , arr [ min_r ] ) ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : swap ( & a [ lo ++ ] , & a [ mid ++ ] ) ; break ; case 1 : mid ++ ; break ; case 2 : swap ( & a [ mid ] , & a [ hi -- ] ) ; break ; } } }
void printArray ( int arr [ ] , int arr_size ) { int i ; for ( i = 0 ; i < arr_size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " n " ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; sort012 ( arr , arr_size ) ; printf ( " array ▁ after ▁ segregation ▁ " ) ; printArray ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
int findNumberOfTriangles ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( arr [ 0 ] ) , comp ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
int main ( ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Total ▁ number ▁ of ▁ triangles ▁ possible ▁ is ▁ % d ▁ " , findNumberOfTriangles ( arr , size ) ) ; return 0 ; }
#include <stdio.h>
int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ;
if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; key = 10 ; printf ( " Index : ▁ % d STRNEWLINE " , binarySearch ( arr , 0 , n - 1 , key ) ) ; return 0 ; }
#include <stdio.h>
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d " , equilibrium ( arr , arr_size ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return -1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 20 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) printf ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " , x ) ; else printf ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " , x , arr [ index ] ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int NEW_LINE int findCandidate ( int * , int ) ; bool isMajority ( int * , int , int ) ;
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
bool isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) if ( a [ i ] == cand ) count ++ ; if ( count > size / 2 ) return 1 ; else return 0 ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) printf ( " ▁ % d ▁ " , cand ) ; else printf ( " No ▁ Majority ▁ Element " ) ; }
int main ( ) { int a [ ] = { 1 , 3 , 3 , 1 , 2 } ; int size = ( sizeof ( a ) ) / sizeof ( a [ 0 ] ) ;
printMajority ( a , size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
void printRepeating ( int arr [ ] , int size ) { int i , j ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) for ( j = i + 1 ; j < size ; j ++ ) if ( arr [ i ] == arr [ j ] ) printf ( " ▁ % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #include <math.h>
int fact ( int n ) ; void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; printf ( " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d " , x , y ) ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) {
int xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) xor ^= i ;
set_bit_no = xor & ~ ( xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} printf ( " n ▁ The ▁ two ▁ repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d ▁ " , x , y ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
void printRepeating ( int arr [ ] , int size ) { int i ; printf ( " The repeating elements are " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) ] > 0 ) arr [ abs ( arr [ i ] ) ] = - arr [ abs ( arr [ i ] ) ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return -1 ; }
int main ( ) { int arr [ 10 ] = { -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Fixed ▁ Point ▁ is ▁ % d " , binarySearch ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
bool find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { printf ( " Triplet ▁ is ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] ) ; return true ; } } } }
return false ; }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; find3Numbers ( A , arr_size , sum ) ; return 0 ; }
#include <stdio.h> NEW_LINE int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int result = search ( arr , n , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
#include <stdio.h>
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE #define RANGE  255
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; printf ( " Sorted ▁ character ▁ array ▁ is ▁ % sn " , arr ) ; return 0 ; }
#include <stdio.h>
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; printf ( " Following ▁ activities ▁ are ▁ selected ▁ n " ) ;
i = 0 ; printf ( " % d ▁ " , i ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { printf ( " % d ▁ " , j ) ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ;
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int minCost ( int cost [ R ] [ C ] , int m , int n ) { if ( n < 0 m < 0 ) return INT_MAX ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ; int minCost ( int cost [ R ] [ C ] , int m , int n ) { int i , j ;
int tc [ R ] [ C ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h>
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
#include <stdio.h>
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h>
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdbool.h> NEW_LINE #include <stdio.h>
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
bool findPartiion ( int arr [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
int main ( ) { int arr [ ] = { 3 , 1 , 5 , 9 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) printf ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ " " of ▁ equal ▁ sum " ) ; else printf ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets " " ▁ of ▁ equal ▁ sum " ) ; return 0 ; }
#include <stdio.h>
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) printf ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ % d ▁ keystrokes ▁ is ▁ % d STRNEWLINE " , N , findoptimal ( N ) ) ; }
#include <stdio.h> NEW_LINE #include <string.h>
void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
#include <stdio.h> NEW_LINE float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; printf ( " % f " , power ( x , y ) ) ; return 0 ; }
#include <stdio.h>
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = -1 , m2 = -1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) { m1 = m2 ;
m2 = ar1 [ i ] ; i ++ ; } else { m1 = m2 ;
m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
int main ( ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) printf ( " Median ▁ is ▁ % d " , getMedian ( ar1 , ar2 , n1 ) ) ; else printf ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ of ▁ unequal ▁ size " ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
int main ( ) { printf ( " % d " , multiply ( 5 , -11 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
int main ( ) { printf ( " % d " , pow ( 5 , 3 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h>
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
int findSmallerInRight ( char * str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; int i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
int main ( ) { char str [ ] = " string " ; printf ( " % d " , findRank ( str ) ) ; return 0 ; }
#include <stdio.h>
int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int main ( ) { int n = 8 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
printf ( " % d ▁ " , C ) ; C = C * ( line - i ) / i ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
#include <stdio.h>
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; printf ( " e ^ x ▁ = ▁ % f " , exponential ( n , x ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) ;
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) printf ( " % d ▁ " , data [ j ] ) ; printf ( " STRNEWLINE " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdlib.h>
int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) printf ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
int hour_angle = 0.5 * ( h * 60 + m ) ; int minute_angle = 6 * m ;
int angle = abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
int main ( ) { printf ( " % d ▁ n " , calcAngle ( 9 , 60 ) ) ; printf ( " % d ▁ n " , calcAngle ( 3 , 30 ) ) ; return 0 ; }
#include <stdio.h>
int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define INT_SIZE  32 NEW_LINE int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
int main ( ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int smallest ( int x , int y , int z ) { int c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int addOne ( int x ) { return ( - ( ~ x ) ) ; }
int main ( ) { printf ( " % d " , addOne ( 13 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int
bool isPowerOfFour ( unsigned int n ) { int count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
#include <stdio.h>
int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
#include <stdio.h>
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; printf ( " % d " , countSetBits ( i ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; printf ( " % d STRNEWLINE " , countSetBitsRec ( num ) ) ; }
#include <stdio.h> NEW_LINE unsigned int nextPowerOf2 ( unsigned int n ) { unsigned int p = 1 ; if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( p < n ) p <<= 1 ; return p ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
#include <stdio.h>
unsigned int nextPowerOf2 ( unsigned int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
# include <stdio.h> NEW_LINE # define bool  int
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; printf ( " Parity ▁ of ▁ no ▁ % d ▁ = ▁ % s " , n , ( getParity ( n ) ? " odd " : " even " ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h> NEW_LINE #include <math.h>
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define bool  int
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
#include <stdio.h>
unsigned int swapBits ( unsigned int x ) {
unsigned int even_bits = x & 0xAAAAAAAA ;
unsigned int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
int main ( ) {
unsigned int x = 23 ;
printf ( " % u ▁ " , swapBits ( x ) ) ; return 0 ; }
#include <stdio.h>
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned i = 1 , pos = 1 ;
while ( ! ( i & n ) ) {
i = i << 1 ;
++ pos ; } return pos ; }
int main ( void ) { int n = 16 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; return 0 ; }
#include <stdio.h>
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; printf ( " Array ▁ after ▁ segregation ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
void nextGreatest ( int arr [ ] , int size ) {
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = -1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; nextGreatest ( arr , size ) ; printf ( " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ) ; printArray ( arr , size ) ; return ( 0 ) ; }
#include <stdio.h>
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; int i , j ; for ( i = 0 ; i < arr_size ; i ++ ) { for ( j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
int main ( ) { int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ;
printf ( " Maximum ▁ difference ▁ is ▁ % d " , maxDiff ( arr , 5 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ maximum ▁ element ▁ is ▁ % d " , findMaximum ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
int getMissingNo ( int a [ ] , int n ) { int i , total ; total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int a [ ] = { 1 , 2 , 4 , 5 , 6 } ; int miss = getMissingNo ( a , 5 ) ; printf ( " % d " , miss ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void printTwoElements ( int arr [ ] , int size ) { int i ; printf ( " The repeating element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } printf ( " and the missing element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) printf ( " % d " , i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; return 0 ; }
#include <stdio.h>
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } printf ( " The two ODD elements are % d & % d " }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h>
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { printf ( " Pair ▁ Found : ▁ ( % d , ▁ % d ) " , arr [ i ] , arr [ j ] ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } printf ( " No ▁ such ▁ pair " ) ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
#include <stdio.h>
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) printf ( " % d , ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] , A [ l ] ) ; } } } }
int main ( ) { int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int X = 91 ; findFourElements ( A , n , X ) ; return 0 ; }
#include <stdio.h>
const int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
void Kroneckerproduct ( int A [ ] [ cola ] , int B [ ] [ colb ] ) { int C [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; printf ( " % d TABSYMBOL " , C [ i + l + 1 ] [ j + k + 1 ] ) ; } } printf ( " STRNEWLINE " ) ; } } }
int main ( ) { int A [ 3 ] [ 2 ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } , B [ 2 ] [ 3 ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; return 0 ; }
#include <stdio.h> NEW_LINE int Identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) printf ( " % d ▁ " , 1 ) ; else printf ( " % d ▁ " , 0 ) ; } printf ( " STRNEWLINE " ) ; } return 0 ; }
int main ( ) { int size = 5 ; identity ( size ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define N  4
void subtract ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; printf ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) printf ( " % d ▁ " , C [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return 0 ; }
void printNGE ( int arr [ ] , int n ) { int next , i , j ; for ( i = 0 ; i < n ; i ++ ) { next = -1 ; for ( j = i + 1 ; j < n ; j ++ ) { if ( arr [ i ] < arr [ j ] ) { next = arr [ j ] ; break ; } } printf ( " % d ▁ - - ▁ % dn " , arr [ i ] , next ) ; } }
int main ( ) { int arr [ ] = { 11 , 13 , 21 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printNGE ( arr , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMin ( int arr [ ] , int low , int high ) {
if ( high < low ) return arr [ 0 ] ;
if ( high == low ) return arr [ low ] ;
int mid = low + ( high - low ) / 2 ;
if ( mid < high && arr [ mid + 1 ] < arr [ mid ] ) return arr [ mid + 1 ] ;
if ( mid > low && arr [ mid ] < arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ high ] > arr [ mid ] ) return findMin ( arr , low , mid - 1 ) ; return findMin ( arr , mid + 1 , high ) ; }
int main ( ) { int arr1 [ ] = { 5 , 6 , 1 , 2 , 3 , 4 } ; int n1 = sizeof ( arr1 ) / sizeof ( arr1 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr1 , 0 , n1 - 1 ) ) ; int arr2 [ ] = { 1 , 2 , 3 , 4 } ; int n2 = sizeof ( arr2 ) / sizeof ( arr2 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr2 , 0 , n2 - 1 ) ) ; int arr3 [ ] = { 1 } ; int n3 = sizeof ( arr3 ) / sizeof ( arr3 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr3 , 0 , n3 - 1 ) ) ; int arr4 [ ] = { 1 , 2 } ; int n4 = sizeof ( arr4 ) / sizeof ( arr4 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr4 , 0 , n4 - 1 ) ) ; int arr5 [ ] = { 2 , 1 } ; int n5 = sizeof ( arr5 ) / sizeof ( arr5 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr5 , 0 , n5 - 1 ) ) ; int arr6 [ ] = { 5 , 6 , 7 , 1 , 2 , 3 , 4 } ; int n6 = sizeof ( arr6 ) / sizeof ( arr6 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr6 , 0 , n6 - 1 ) ) ; int arr7 [ ] = { 1 , 2 , 3 , 4 , 5 , 6 , 7 } ; int n7 = sizeof ( arr7 ) / sizeof ( arr7 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr7 , 0 , n7 - 1 ) ) ; int arr8 [ ] = { 2 , 3 , 4 , 5 , 6 , 7 , 8 , 1 } ; int n8 = sizeof ( arr8 ) / sizeof ( arr8 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr8 , 0 , n8 - 1 ) ) ; int arr9 [ ] = { 3 , 4 , 5 , 1 , 2 } ; int n9 = sizeof ( arr9 ) / sizeof ( arr9 [ 0 ] ) ; printf ( " The ▁ minimum ▁ element ▁ is ▁ % d STRNEWLINE " , findMin ( arr9 , 0 , n9 - 1 ) ) ; return 0 ; }
void print2largest ( int arr [ ] , int arr_size ) { int i , first , second ;
if ( arr_size < 2 ) { printf ( " ▁ Invalid ▁ Input ▁ " ) ; return ; } first = second = INT_MIN ; for ( i = 0 ; i < arr_size ; i ++ ) {
if ( arr [ i ] > first ) { second = first ; first = arr [ i ] ; }
else if ( arr [ i ] > second && arr [ i ] != first ) second = arr [ i ] ; } if ( second == INT_MIN ) printf ( " There ▁ is ▁ no ▁ second ▁ largest ▁ element STRNEWLINE " ) ; else printf ( " The ▁ second ▁ largest ▁ element ▁ is ▁ % dn " , second ) ; }
int main ( ) { int arr [ ] = { 12 , 35 , 1 , 10 , 34 , 1 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; print2largest ( arr , n ) ; return 0 ; }
int FindMaxSum ( int arr [ ] , int n ) { int incl = arr [ 0 ] ; int excl = 0 ; int excl_new ; int i ; for ( i = 1 ; i < n ; i ++ ) {
excl_new = ( incl > excl ) ? incl : excl ;
incl = excl + arr [ i ] ; excl = excl_new ; }
return ( ( incl > excl ) ? incl : excl ) ; }
int main ( ) { int arr [ ] = { 5 , 5 , 10 , 100 , 10 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d ▁ n " , FindMaxSum ( arr , n ) ) ; return 0 ; }
int minJumps ( int arr [ ] , int l , int h ) {
if ( h == l ) return 0 ;
if ( arr [ l ] == 0 ) return INT_MAX ;
int min = INT_MAX ; for ( int i = l + 1 ; i <= h && i <= l + arr [ l ] ; i ++ ) { int jumps = minJumps ( arr , i , h ) ; if ( jumps != INT_MAX && jumps + 1 < min ) min = jumps + 1 ; } return min ; }
int main ( ) { int arr [ ] = { 1 , 3 , 6 , 3 , 2 , 3 , 6 , 8 , 9 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Minimum ▁ number ▁ of ▁ jumps ▁ to ▁ reach ▁ end ▁ is ▁ % d ▁ " , minJumps ( arr , 0 , n - 1 ) ) ; return 0 ; }
int maxSumIS ( int arr [ ] , int n ) { int i , j , max = 0 ; int msis [ n ] ;
for ( i = 0 ; i < n ; i ++ ) msis [ i ] = arr [ i ] ;
for ( i = 1 ; i < n ; i ++ ) for ( j = 0 ; j < i ; j ++ ) if ( arr [ i ] > arr [ j ] && msis [ i ] < msis [ j ] + arr [ i ] ) msis [ i ] = msis [ j ] + arr [ i ] ;
for ( i = 0 ; i < n ; i ++ ) if ( max < msis [ i ] ) max = msis [ i ] ; return max ; }
int main ( ) { int arr [ ] = { 1 , 101 , 2 , 3 , 100 , 4 , 5 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Sum ▁ of ▁ maximum ▁ sum ▁ increasing ▁ " " subsequence ▁ is ▁ % d STRNEWLINE " , maxSumIS ( arr , n ) ) ; return 0 ; }
#define NA  -1
void moveToEnd ( int mPlusN [ ] , int size ) { int i = 0 , j = size - 1 ; for ( i = size - 1 ; i >= 0 ; i -- ) if ( mPlusN [ i ] != NA ) { mPlusN [ j ] = mPlusN [ i ] ; j -- ; } }
int merge ( int mPlusN [ ] , int N [ ] , int m , int n ) { int i = n ;
int j = 0 ;
int k = 0 ;
while ( k < ( m + n ) ) {
if ( ( j == n ) || ( i < ( m + n ) && mPlusN [ i ] <= N [ j ] ) ) { mPlusN [ k ] = mPlusN [ i ] ; k ++ ; i ++ ; }
else { mPlusN [ k ] = N [ j ] ; k ++ ; j ++ ; } } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int mPlusN [ ] = { 2 , 8 , NA , NA , NA , 13 , NA , 15 , 20 } ; int N [ ] = { 5 , 7 , 9 , 25 } ; int n = sizeof ( N ) / sizeof ( N [ 0 ] ) ; int m = sizeof ( mPlusN ) / sizeof ( mPlusN [ 0 ] ) - n ;
moveToEnd ( mPlusN , m + n ) ;
merge ( mPlusN , N , m , n ) ;
printArray ( mPlusN , m + n ) ; return 0 ; }
# include <stdio.h> NEW_LINE # include <stdlib.h> NEW_LINE # include <math.h> NEW_LINE void minAbsSumPair ( int arr [ ] , int arr_size ) { int inv_count = 0 ; int l , r , min_sum , sum , min_l , min_r ;
if ( arr_size < 2 ) { printf ( " Invalid ▁ Input " ) ; return ; }
min_l = 0 ; min_r = 1 ; min_sum = arr [ 0 ] + arr [ 1 ] ; for ( l = 0 ; l < arr_size - 1 ; l ++ ) { for ( r = l + 1 ; r < arr_size ; r ++ ) { sum = arr [ l ] + arr [ r ] ; if ( abs ( min_sum ) > abs ( sum ) ) { min_sum = sum ; min_l = l ; min_r = r ; } } } printf ( " ▁ The ▁ two ▁ elements ▁ whose ▁ sum ▁ is ▁ minimum ▁ are ▁ % d ▁ and ▁ % d " , arr [ min_l ] , arr [ min_r ] ) ; }
int main ( ) { int arr [ ] = { 1 , 60 , -10 , 70 , -80 , 85 } ; minAbsSumPair ( arr , 6 ) ; getchar ( ) ; return 0 ; }
void swap ( int * a , int * b ) { int temp = * a ; * a = * b ; * b = temp ; }
void sort012 ( int a [ ] , int arr_size ) { int lo = 0 ; int hi = arr_size - 1 ; int mid = 0 ; while ( mid <= hi ) { switch ( a [ mid ] ) { case 0 : swap ( & a [ lo ++ ] , & a [ mid ++ ] ) ; break ; case 1 : mid ++ ; break ; case 2 : swap ( & a [ mid ] , & a [ hi -- ] ) ; break ; } } }
void printArray ( int arr [ ] , int arr_size ) { int i ; for ( i = 0 ; i < arr_size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " n " ) ; }
int main ( ) { int arr [ ] = { 0 , 1 , 1 , 0 , 1 , 2 , 1 , 2 , 0 , 0 , 0 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int i ; sort012 ( arr , arr_size ) ; printf ( " array ▁ after ▁ segregation ▁ " ) ; printArray ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
int findNumberOfTriangles ( int arr [ ] , int n ) {
qsort ( arr , n , sizeof ( arr [ 0 ] ) , comp ) ;
int count = 0 ;
for ( int i = 0 ; i < n - 2 ; ++ i ) {
int k = i + 2 ;
for ( int j = i + 1 ; j < n ; ++ j ) {
while ( k < n && arr [ i ] + arr [ j ] > arr [ k ] ) ++ k ;
if ( k > j ) count += k - j - 1 ; } } return count ; }
int main ( ) { int arr [ ] = { 10 , 21 , 22 , 100 , 101 , 200 , 300 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Total ▁ number ▁ of ▁ triangles ▁ possible ▁ is ▁ % d ▁ " , findNumberOfTriangles ( arr , size ) ) ; return 0 ; }
int binarySearch ( int arr [ ] , int low , int high , int key ) { if ( high < low ) return -1 ; int mid = ( low + high ) / 2 ;
if ( key == arr [ mid ] ) return mid ; if ( key > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high , key ) ; return binarySearch ( arr , low , ( mid - 1 ) , key ) ; }
int main ( ) { int arr [ ] = { 5 , 6 , 7 , 8 , 9 , 10 } ; int n , key ; n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; key = 10 ; printf ( " Index : ▁ % d STRNEWLINE " , binarySearch ( arr , 0 , n - 1 , key ) ) ; return 0 ; }
int equilibrium ( int arr [ ] , int n ) { int i , j ; int leftsum , rightsum ;
for ( i = 0 ; i < n ; ++ i ) { leftsum = 0 ; rightsum = 0 ;
for ( j = 0 ; j < i ; j ++ ) leftsum += arr [ j ] ;
for ( j = i + 1 ; j < n ; j ++ ) rightsum += arr [ j ] ;
if ( leftsum == rightsum ) return i ; }
return -1 ; }
int main ( ) { int arr [ ] = { -7 , 1 , 5 , 2 , -4 , 3 , 0 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " % d " , equilibrium ( arr , arr_size ) ) ; getchar ( ) ; return 0 ; }
int ceilSearch ( int arr [ ] , int low , int high , int x ) { int mid ;
if ( x <= arr [ low ] ) return low ;
if ( x > arr [ high ] ) return -1 ;
mid = ( low + high ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
else if ( arr [ mid ] < x ) { if ( mid + 1 <= high && x <= arr [ mid + 1 ] ) return mid + 1 ; else return ceilSearch ( arr , mid + 1 , high , x ) ; }
else { if ( mid - 1 >= low && x > arr [ mid - 1 ] ) return mid ; else return ceilSearch ( arr , low , mid - 1 , x ) ; } }
int main ( ) { int arr [ ] = { 1 , 2 , 8 , 10 , 10 , 12 , 19 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 20 ; int index = ceilSearch ( arr , 0 , n - 1 , x ) ; if ( index == -1 ) printf ( " Ceiling ▁ of ▁ % d ▁ doesn ' t ▁ exist ▁ in ▁ array ▁ " , x ) ; else printf ( " ceiling ▁ of ▁ % d ▁ is ▁ % d " , x , arr [ index ] ) ; getchar ( ) ; return 0 ; }
int findCandidate ( int a [ ] , int size ) { int maj_index = 0 , count = 1 ; int i ; for ( i = 1 ; i < size ; i ++ ) { if ( a [ maj_index ] == a [ i ] ) count ++ ; else count -- ; if ( count == 0 ) { maj_index = i ; count = 1 ; } } return a [ maj_index ] ; }
bool isMajority ( int a [ ] , int size , int cand ) { int i , count = 0 ; for ( i = 0 ; i < size ; i ++ ) if ( a [ i ] == cand ) count ++ ; if ( count > size / 2 ) return 1 ; else return 0 ; }
void printMajority ( int a [ ] , int size ) {
int cand = findCandidate ( a , size ) ;
if ( isMajority ( a , size , cand ) ) printf ( " ▁ % d ▁ " , cand ) ; else printf ( " No ▁ Majority ▁ Element " ) ; }
int main ( ) { int a [ ] = { 1 , 3 , 3 , 1 , 2 } ; int size = ( sizeof ( a ) ) / sizeof ( a [ 0 ] ) ;
printMajority ( a , size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) { int i , j ; printf ( " ▁ Repeating ▁ elements ▁ are ▁ " ) ; for ( i = 0 ; i < size ; i ++ ) for ( j = i + 1 ; j < size ; j ++ ) if ( arr [ i ] == arr [ j ] ) printf ( " ▁ % d ▁ " , arr [ i ] ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
int fact ( int n ) ; void printRepeating ( int arr [ ] , int size ) {
int S = 0 ;
int P = 1 ;
int x , y ;
int D ; int n = size - 2 , i ;
for ( i = 0 ; i < size ; i ++ ) { S = S + arr [ i ] ; P = P * arr [ i ] ; }
S = S - n * ( n + 1 ) / 2 ;
P = P / fact ( n ) ;
D = sqrt ( S * S - 4 * P ) ; x = ( D + S ) / 2 ; y = ( S - D ) / 2 ; printf ( " The ▁ two ▁ Repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d " , x , y ) ; }
int fact ( int n ) { return ( n == 0 ) ? 1 : n * fact ( n - 1 ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) {
int xor = arr [ 0 ] ;
int set_bit_no ; int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor ^= arr [ i ] ; for ( i = 1 ; i <= n ; i ++ ) xor ^= i ;
set_bit_no = xor & ~ ( xor - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ;
} for ( i = 1 ; i <= n ; i ++ ) { if ( i & set_bit_no ) x = x ^ i ;
else y = y ^ i ;
} printf ( " n ▁ The ▁ two ▁ repeating ▁ elements ▁ are ▁ % d ▁ & ▁ % d ▁ " , x , y ) ; }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
void printRepeating ( int arr [ ] , int size ) { int i ; printf ( " The repeating elements are " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) ] > 0 ) arr [ abs ( arr [ i ] ) ] = - arr [ abs ( arr [ i ] ) ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printRepeating ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int binarySearch ( int arr [ ] , int low , int high ) { if ( high >= low ) { int mid = ( low + high ) / 2 ;
if ( mid == arr [ mid ] ) return mid ; if ( mid > arr [ mid ] ) return binarySearch ( arr , ( mid + 1 ) , high ) ; else return binarySearch ( arr , low , ( mid - 1 ) ) ; }
return -1 ; }
int main ( ) { int arr [ 10 ] = { -10 , -1 , 0 , 3 , 10 , 11 , 30 , 50 , 100 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " Fixed ▁ Point ▁ is ▁ % d " , binarySearch ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
bool find3Numbers ( int A [ ] , int arr_size , int sum ) { int l , r ;
for ( int i = 0 ; i < arr_size - 2 ; i ++ ) {
for ( int j = i + 1 ; j < arr_size - 1 ; j ++ ) {
for ( int k = j + 1 ; k < arr_size ; k ++ ) { if ( A [ i ] + A [ j ] + A [ k ] == sum ) { printf ( " Triplet ▁ is ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] ) ; return true ; } } } }
return false ; }
int main ( ) { int A [ ] = { 1 , 4 , 45 , 6 , 10 , 8 } ; int sum = 22 ; int arr_size = sizeof ( A ) / sizeof ( A [ 0 ] ) ; find3Numbers ( A , arr_size , sum ) ; return 0 ; }
#include <stdio.h> NEW_LINE int search ( int arr [ ] , int n , int x ) { int i ; for ( i = 0 ; i < n ; i ++ ) if ( arr [ i ] == x ) return i ; return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int x = 10 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
int result = search ( arr , n , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
int binarySearch ( int arr [ ] , int l , int r , int x ) { if ( r >= l ) { int mid = l + ( r - l ) / 2 ;
if ( arr [ mid ] == x ) return mid ;
if ( arr [ mid ] > x ) return binarySearch ( arr , l , mid - 1 , x ) ;
return binarySearch ( arr , mid + 1 , r , x ) ; }
return -1 ; }
int main ( void ) { int arr [ ] = { 2 , 3 , 4 , 10 , 40 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int x = 10 ; int result = binarySearch ( arr , 0 , n - 1 , x ) ; ( result == -1 ) ? printf ( " Element ▁ is ▁ not ▁ present ▁ in ▁ array " ) : printf ( " Element ▁ is ▁ present ▁ at ▁ index ▁ % d " , result ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <string.h> NEW_LINE #define RANGE  255
void countSort ( char arr [ ] ) {
char output [ strlen ( arr ) ] ;
int count [ RANGE + 1 ] , i ; memset ( count , 0 , sizeof ( count ) ) ;
for ( i = 0 ; arr [ i ] ; ++ i ) ++ count [ arr [ i ] ] ;
for ( i = 1 ; i <= RANGE ; ++ i ) count [ i ] += count [ i - 1 ] ;
for ( i = 0 ; arr [ i ] ; ++ i ) { output [ count [ arr [ i ] ] - 1 ] = arr [ i ] ; -- count [ arr [ i ] ] ; }
for ( i = 0 ; arr [ i ] ; ++ i ) arr [ i ] = output [ i ] ; }
int main ( ) { char arr [ ] = " geeksforgeeks " ; countSort ( arr ) ; printf ( " Sorted ▁ character ▁ array ▁ is ▁ % sn " , arr ) ; return 0 ; }
void printMaxActivities ( int s [ ] , int f [ ] , int n ) { int i , j ; printf ( " Following ▁ activities ▁ are ▁ selected ▁ n " ) ;
i = 0 ; printf ( " % d ▁ " , i ) ;
for ( j = 1 ; j < n ; j ++ ) {
if ( s [ j ] >= f [ i ] ) { printf ( " % d ▁ " , j ) ; i = j ; } } }
int main ( ) { int s [ ] = { 1 , 3 , 0 , 5 , 8 , 5 } ; int f [ ] = { 2 , 4 , 6 , 7 , 9 , 9 } ; int n = sizeof ( s ) / sizeof ( s [ 0 ] ) ; printMaxActivities ( s , f , n ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ;
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int minCost ( int cost [ R ] [ C ] , int m , int n ) { if ( n < 0 m < 0 ) return INT_MAX ; else if ( m == 0 && n == 0 ) return cost [ m ] [ n ] ; else return cost [ m ] [ n ] + min ( minCost ( cost , m - 1 , n - 1 ) , minCost ( cost , m - 1 , n ) , minCost ( cost , m , n - 1 ) ) ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <limits.h> NEW_LINE #define R  3 NEW_LINE #define C  3 NEW_LINE int min ( int x , int y , int z ) ; int minCost ( int cost [ R ] [ C ] , int m , int n ) { int i , j ;
int tc [ R ] [ C ] ; tc [ 0 ] [ 0 ] = cost [ 0 ] [ 0 ] ;
for ( i = 1 ; i <= m ; i ++ ) tc [ i ] [ 0 ] = tc [ i - 1 ] [ 0 ] + cost [ i ] [ 0 ] ;
for ( j = 1 ; j <= n ; j ++ ) tc [ 0 ] [ j ] = tc [ 0 ] [ j - 1 ] + cost [ 0 ] [ j ] ;
for ( i = 1 ; i <= m ; i ++ ) for ( j = 1 ; j <= n ; j ++ ) tc [ i ] [ j ] = min ( tc [ i - 1 ] [ j - 1 ] , tc [ i - 1 ] [ j ] , tc [ i ] [ j - 1 ] ) + cost [ i ] [ j ] ; return tc [ m ] [ n ] ; }
int min ( int x , int y , int z ) { if ( x < y ) return ( x < z ) ? x : z ; else return ( y < z ) ? y : z ; }
int main ( ) { int cost [ R ] [ C ] = { { 1 , 2 , 3 } , { 4 , 8 , 2 } , { 1 , 5 , 3 } } ; printf ( " ▁ % d ▁ " , minCost ( cost , 2 , 2 ) ) ; return 0 ; }
int binomialCoeff ( int n , int k ) {
if ( k > n ) return 0 ; if ( k == 0 k == n ) return 1 ;
return binomialCoeff ( n - 1 , k - 1 ) + binomialCoeff ( n - 1 , k ) ; }
int main ( ) { int n = 5 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
int knapSack ( int W , int wt [ ] , int val [ ] , int n ) { int i , w ; int K [ n + 1 ] [ W + 1 ] ;
for ( i = 0 ; i <= n ; i ++ ) { for ( w = 0 ; w <= W ; w ++ ) { if ( i == 0 w == 0 ) K [ i ] [ w ] = 0 ; else if ( wt [ i - 1 ] <= w ) K [ i ] [ w ] = max ( val [ i - 1 ] + K [ i - 1 ] [ w - wt [ i - 1 ] ] , K [ i - 1 ] [ w ] ) ; else K [ i ] [ w ] = K [ i - 1 ] [ w ] ; } } return K [ n ] [ W ] ; }
int main ( ) { int val [ ] = { 60 , 100 , 120 } ; int wt [ ] = { 10 , 20 , 30 } ; int W = 50 ; int n = sizeof ( val ) / sizeof ( val [ 0 ] ) ; printf ( " % d " , knapSack ( W , wt , val , n ) ) ; return 0 ; }
int lps ( char * seq , int i , int j ) {
if ( i == j ) return 1 ;
if ( seq [ i ] == seq [ j ] && i + 1 == j ) return 2 ;
if ( seq [ i ] == seq [ j ] ) return lps ( seq , i + 1 , j - 1 ) + 2 ;
return max ( lps ( seq , i , j - 1 ) , lps ( seq , i + 1 , j ) ) ; }
int main ( ) { char seq [ ] = " GEEKSFORGEEKS " ; int n = strlen ( seq ) ; printf ( " The ▁ length ▁ of ▁ the ▁ LPS ▁ is ▁ % d " , lps ( seq , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
bool isSubsetSum ( int arr [ ] , int n , int sum ) {
if ( sum == 0 ) return true ; if ( n == 0 && sum != 0 ) return false ;
if ( arr [ n - 1 ] > sum ) return isSubsetSum ( arr , n - 1 , sum ) ;
return isSubsetSum ( arr , n - 1 , sum ) || isSubsetSum ( arr , n - 1 , sum - arr [ n - 1 ] ) ; }
bool findPartiion ( int arr [ ] , int n ) {
int sum = 0 ; for ( int i = 0 ; i < n ; i ++ ) sum += arr [ i ] ;
if ( sum % 2 != 0 ) return false ;
return isSubsetSum ( arr , n , sum / 2 ) ; }
int main ( ) { int arr [ ] = { 3 , 1 , 5 , 9 , 12 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ;
if ( findPartiion ( arr , n ) == true ) printf ( " Can ▁ be ▁ divided ▁ into ▁ two ▁ subsets ▁ " " of ▁ equal ▁ sum " ) ; else printf ( " Can ▁ not ▁ be ▁ divided ▁ into ▁ two ▁ subsets " " ▁ of ▁ equal ▁ sum " ) ; return 0 ; }
int findoptimal ( int N ) {
if ( N <= 6 ) return N ;
int max = 0 ;
int b ; for ( b = N - 3 ; b >= 1 ; b -- ) {
int curr = ( N - b - 1 ) * findoptimal ( b ) ; if ( curr > max ) max = curr ; } return max ; }
int main ( ) { int N ;
for ( N = 1 ; N <= 20 ; N ++ ) printf ( " Maximum ▁ Number ▁ of ▁ A ' s ▁ with ▁ % d ▁ keystrokes ▁ is ▁ % d STRNEWLINE " , N , findoptimal ( N ) ) ; }
void search ( char pat [ ] , char txt [ ] ) { int M = strlen ( pat ) ; int N = strlen ( txt ) ; int i = 0 ; while ( i <= N - M ) { int j ;
for ( j = 0 ; j < M ; j ++ ) if ( txt [ i + j ] != pat [ j ] ) break ;
if ( j == M ) { printf ( " Pattern ▁ found ▁ at ▁ index ▁ % d ▁ STRNEWLINE " , i ) ; i = i + M ; } else if ( j == 0 ) i = i + 1 ; else
i = i + j ; } }
int main ( ) { char txt [ ] = " ABCEABCDABCEABCD " ; char pat [ ] = " ABCD " ; search ( pat , txt ) ; return 0 ; }
#include <stdio.h> NEW_LINE float power ( float x , int y ) { float temp ; if ( y == 0 ) return 1 ; temp = power ( x , y / 2 ) ; if ( y % 2 == 0 ) return temp * temp ; else { if ( y > 0 ) return x * temp * temp ; else return ( temp * temp ) / x ; } }
int main ( ) { float x = 2 ; int y = -3 ; printf ( " % f " , power ( x , y ) ) ; return 0 ; }
int getMedian ( int ar1 [ ] , int ar2 [ ] , int n ) { int i = 0 ; int j = 0 ; int count ; int m1 = -1 , m2 = -1 ;
for ( count = 0 ; count <= n ; count ++ ) {
if ( i == n ) { m1 = m2 ; m2 = ar2 [ 0 ] ; break ; }
else if ( j == n ) { m1 = m2 ; m2 = ar1 [ 0 ] ; break ; }
if ( ar1 [ i ] <= ar2 [ j ] ) { m1 = m2 ;
m2 = ar1 [ i ] ; i ++ ; } else { m1 = m2 ;
m2 = ar2 [ j ] ; j ++ ; } } return ( m1 + m2 ) / 2 ; }
int main ( ) { int ar1 [ ] = { 1 , 12 , 15 , 26 , 38 } ; int ar2 [ ] = { 2 , 13 , 17 , 30 , 45 } ; int n1 = sizeof ( ar1 ) / sizeof ( ar1 [ 0 ] ) ; int n2 = sizeof ( ar2 ) / sizeof ( ar2 [ 0 ] ) ; if ( n1 == n2 ) printf ( " Median ▁ is ▁ % d " , getMedian ( ar1 , ar2 , n1 ) ) ; else printf ( " Doesn ' t ▁ work ▁ for ▁ arrays ▁ of ▁ unequal ▁ size " ) ; getchar ( ) ; return 0 ; }
int multiply ( int x , int y ) {
if ( y == 0 ) return 0 ;
if ( y > 0 ) return ( x + multiply ( x , y - 1 ) ) ;
if ( y < 0 ) return - multiply ( x , - y ) ; }
int main ( ) { printf ( " % d " , multiply ( 5 , -11 ) ) ; getchar ( ) ; return 0 ; }
int pow ( int a , int b ) { if ( b == 0 ) return 1 ; int answer = a ; int increment = a ; int i , j ; for ( i = 1 ; i < b ; i ++ ) { for ( j = 1 ; j < a ; j ++ ) { answer += increment ; } increment = answer ; } return answer ; }
int main ( ) { printf ( " % d " , pow ( 5 , 3 ) ) ; getchar ( ) ; return 0 ; }
int fact ( int n ) { return ( n <= 1 ) ? 1 : n * fact ( n - 1 ) ; }
int findSmallerInRight ( char * str , int low , int high ) { int countRight = 0 , i ; for ( i = low + 1 ; i <= high ; ++ i ) if ( str [ i ] < str [ low ] ) ++ countRight ; return countRight ; }
int findRank ( char * str ) { int len = strlen ( str ) ; int mul = fact ( len ) ; int rank = 1 ; int countRight ; int i ; for ( i = 0 ; i < len ; ++ i ) { mul /= len - i ;
countRight = findSmallerInRight ( str , i , len - 1 ) ; rank += countRight * mul ; } return rank ; }
int main ( ) { char str [ ] = " string " ; printf ( " % d " , findRank ( str ) ) ; return 0 ; }
int binomialCoeff ( int n , int k ) { int res = 1 ;
if ( k > n - k ) k = n - k ;
for ( int i = 0 ; i < k ; ++ i ) { res *= ( n - i ) ; res /= ( i + 1 ) ; } return res ; }
int main ( ) { int n = 8 , k = 2 ; printf ( " Value ▁ of ▁ C ( % d , ▁ % d ) ▁ is ▁ % d ▁ " , n , k , binomialCoeff ( n , k ) ) ; return 0 ; }
void printPascal ( int n ) { for ( int line = 1 ; line <= n ; line ++ ) {
int C = 1 ; for ( int i = 1 ; i <= line ; i ++ ) {
printf ( " % d ▁ " , C ) ; C = C * ( line - i ) / i ; } printf ( " STRNEWLINE " ) ; } }
int main ( ) { int n = 5 ; printPascal ( n ) ; return 0 ; }
float exponential ( int n , float x ) {
float sum = 1.0f ; for ( int i = n - 1 ; i > 0 ; -- i ) sum = 1 + x * sum / i ; return sum ; }
int main ( ) { int n = 10 ; float x = 1.0f ; printf ( " e ^ x ▁ = ▁ % f " , exponential ( n , x ) ) ; return 0 ; }
void printCombination ( int arr [ ] , int n , int r ) {
int data [ r ] ;
combinationUtil ( arr , n , r , 0 , data , 0 ) ; }
void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) {
if ( index == r ) { for ( int j = 0 ; j < r ; j ++ ) printf ( " % d ▁ " , data [ j ] ) ; printf ( " STRNEWLINE " ) ; return ; }
if ( i >= n ) return ;
data [ index ] = arr [ i ] ; combinationUtil ( arr , n , r , index + 1 , data , i + 1 ) ;
combinationUtil ( arr , n , r , index , data , i + 1 ) ; }
int main ( ) { int arr [ ] = { 1 , 2 , 3 , 4 , 5 } ; int r = 3 ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printCombination ( arr , n , r ) ; return 0 ; }
int min ( int x , int y ) { return ( x < y ) ? x : y ; }
int calcAngle ( double h , double m ) {
if ( h < 0 m < 0 h > 12 m > 60 ) printf ( " Wrong ▁ input " ) ; if ( h == 12 ) h = 0 ; if ( m == 60 ) { m = 0 ; h += 1 ; if ( h > 12 ) h = h - 12 ; }
int hour_angle = 0.5 * ( h * 60 + m ) ; int minute_angle = 6 * m ;
int angle = abs ( hour_angle - minute_angle ) ;
angle = min ( 360 - angle , angle ) ; return angle ; }
int main ( ) { printf ( " % d ▁ n " , calcAngle ( 9 , 60 ) ) ; printf ( " % d ▁ n " , calcAngle ( 3 , 30 ) ) ; return 0 ; }
int getSingle ( int arr [ ] , int n ) { int ones = 0 , twos = 0 ; int common_bit_mask ; for ( int i = 0 ; i < n ; i ++ ) {
twos = twos | ( ones & arr [ i ] ) ;
ones = ones ^ arr [ i ] ;
common_bit_mask = ~ ( ones & twos ) ;
ones &= common_bit_mask ;
twos &= common_bit_mask ; } return ones ; }
int main ( ) { int arr [ ] = { 3 , 3 , 2 , 3 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE #define INT_SIZE  32 NEW_LINE int getSingle ( int arr [ ] , int n ) {
int result = 0 ; int x , sum ;
for ( int i = 0 ; i < INT_SIZE ; i ++ ) {
sum = 0 ; x = ( 1 << i ) ; for ( int j = 0 ; j < n ; j ++ ) { if ( arr [ j ] & x ) sum ++ ; }
if ( ( sum % 3 ) != 0 ) result |= x ; } return result ; }
int main ( ) { int arr [ ] = { 12 , 1 , 12 , 3 , 12 , 1 , 1 , 2 , 3 , 2 , 2 , 3 , 7 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ element ▁ with ▁ single ▁ occurrence ▁ is ▁ % d ▁ " , getSingle ( arr , n ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int smallest ( int x , int y , int z ) { int c = 0 ; while ( x && y && z ) { x -- ; y -- ; z -- ; c ++ ; } return c ; }
int main ( ) { int x = 12 , y = 15 , z = 5 ; printf ( " Minimum ▁ of ▁ 3 ▁ numbers ▁ is ▁ % d " , smallest ( x , y , z ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int addOne ( int x ) { return ( - ( ~ x ) ) ; }
int main ( ) { printf ( " % d " , addOne ( 13 ) ) ; getchar ( ) ; return 0 ; }
bool isPowerOfFour ( unsigned int n ) { int count = 0 ;
if ( n && ! ( n & ( n - 1 ) ) ) {
while ( n > 1 ) { n >>= 1 ; count += 1 ; }
return ( count % 2 == 0 ) ? 1 : 0 ; }
return 0 ; }
int main ( ) { int test_no = 64 ; if ( isPowerOfFour ( test_no ) ) printf ( " % d ▁ is ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; else printf ( " % d ▁ is ▁ not ▁ a ▁ power ▁ of ▁ 4" , test_no ) ; getchar ( ) ; }
int min ( int x , int y ) { return y ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int max ( int x , int y ) { return x ^ ( ( x ^ y ) & - ( x < y ) ) ; }
int main ( ) { int x = 15 ; int y = 6 ; printf ( " Minimum ▁ of ▁ % d ▁ and ▁ % d ▁ is ▁ " , x , y ) ; printf ( " % d " , min ( x , y ) ) ; printf ( " Maximum of % d and % d is " printf ( " % d " , max ( x , y ) ) ; getchar ( ) ; }
unsigned int countSetBits ( unsigned int n ) { unsigned int count = 0 ; while ( n ) { count += n & 1 ; n >>= 1 ; } return count ; }
int main ( ) { int i = 9 ; printf ( " % d " , countSetBits ( i ) ) ; return 0 ; }
#include <stdio.h> NEW_LINE int num_to_bits [ 16 ] = { 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 } ;
unsigned int countSetBitsRec ( unsigned int num ) { int nibble = 0 ; if ( 0 == num ) return num_to_bits [ 0 ] ;
nibble = num & 0xf ;
return num_to_bits [ nibble ] + countSetBitsRec ( num >> 4 ) ; }
int main ( ) { int num = 31 ; printf ( " % d STRNEWLINE " , countSetBitsRec ( num ) ) ; }
#include <stdio.h> NEW_LINE unsigned int nextPowerOf2 ( unsigned int n ) { unsigned int p = 1 ; if ( n && ! ( n & ( n - 1 ) ) ) return n ; while ( p < n ) p <<= 1 ; return p ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
unsigned int nextPowerOf2 ( unsigned int n ) { n -- ; n |= n >> 1 ; n |= n >> 2 ; n |= n >> 4 ; n |= n >> 8 ; n |= n >> 16 ; n ++ ; return n ; }
int main ( ) { unsigned int n = 5 ; printf ( " % d " , nextPowerOf2 ( n ) ) ; return 0 ; }
bool getParity ( unsigned int n ) { bool parity = 0 ; while ( n ) { parity = ! parity ; n = n & ( n - 1 ) ; } return parity ; }
int main ( ) { unsigned int n = 7 ; printf ( " Parity ▁ of ▁ no ▁ % d ▁ = ▁ % s " , n , ( getParity ( n ) ? " odd " : " even " ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE #include <stdbool.h> NEW_LINE #include <math.h>
bool isPowerOfTwo ( int n ) { if ( n == 0 ) return false ; return ( ceil ( log2 ( n ) ) == floor ( log2 ( n ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
bool isPowerOfTwo ( int x ) {
return x && ( ! ( x & ( x - 1 ) ) ) ; }
int main ( ) { isPowerOfTwo ( 31 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; isPowerOfTwo ( 64 ) ? printf ( " Yes STRNEWLINE " ) : printf ( " No STRNEWLINE " ) ; return 0 ; }
unsigned int swapBits ( unsigned int x ) {
unsigned int even_bits = x & 0xAAAAAAAA ;
unsigned int odd_bits = x & 0x55555555 ;
even_bits >>= 1 ;
odd_bits <<= 1 ;
return ( even_bits odd_bits ) ; }
unsigned int x = 23 ;
printf ( " % u ▁ " , swapBits ( x ) ) ; return 0 ; }
int isPowerOfTwo ( unsigned n ) { return n && ( ! ( n & ( n - 1 ) ) ) ; }
int findPosition ( unsigned n ) { if ( ! isPowerOfTwo ( n ) ) return -1 ; unsigned i = 1 , pos = 1 ;
while ( ! ( i & n ) ) {
i = i << 1 ;
++ pos ; } return pos ; }
int main ( void ) { int n = 16 ; int pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 12 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; n = 128 ; pos = findPosition ( n ) ; ( pos == -1 ) ? printf ( " n ▁ = ▁ % d , ▁ Invalid ▁ number STRNEWLINE " , n ) : printf ( " n ▁ = ▁ % d , ▁ Position ▁ % d ▁ STRNEWLINE " , n , pos ) ; return 0 ; }
void segregate0and1 ( int arr [ ] , int size ) {
int left = 0 , right = size - 1 ; while ( left < right ) {
while ( arr [ left ] == 0 && left < right ) left ++ ;
while ( arr [ right ] == 1 && left < right ) right -- ;
if ( left < right ) { arr [ left ] = 0 ; arr [ right ] = 1 ; left ++ ; right -- ; } } }
int main ( ) { int arr [ ] = { 0 , 1 , 0 , 1 , 1 , 1 } ; int i , arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; segregate0and1 ( arr , arr_size ) ; printf ( " Array ▁ after ▁ segregation ▁ " ) ; for ( i = 0 ; i < 6 ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; getchar ( ) ; return 0 ; }
void nextGreatest ( int arr [ ] , int size ) {
int max_from_right = arr [ size - 1 ] ;
arr [ size - 1 ] = -1 ;
for ( int i = size - 2 ; i >= 0 ; i -- ) {
int temp = arr [ i ] ;
arr [ i ] = max_from_right ;
if ( max_from_right < temp ) max_from_right = temp ; } }
void printArray ( int arr [ ] , int size ) { int i ; for ( i = 0 ; i < size ; i ++ ) printf ( " % d ▁ " , arr [ i ] ) ; printf ( " STRNEWLINE " ) ; }
int main ( ) { int arr [ ] = { 16 , 17 , 4 , 3 , 5 , 2 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; nextGreatest ( arr , size ) ; printf ( " The ▁ modified ▁ array ▁ is : ▁ STRNEWLINE " ) ; printArray ( arr , size ) ; return ( 0 ) ; }
int maxDiff ( int arr [ ] , int arr_size ) { int max_diff = arr [ 1 ] - arr [ 0 ] ; int i , j ; for ( i = 0 ; i < arr_size ; i ++ ) { for ( j = i + 1 ; j < arr_size ; j ++ ) { if ( arr [ j ] - arr [ i ] > max_diff ) max_diff = arr [ j ] - arr [ i ] ; } } return max_diff ; }
int main ( ) { int arr [ ] = { 1 , 2 , 90 , 10 , 110 } ;
printf ( " Maximum ▁ difference ▁ is ▁ % d " , maxDiff ( arr , 5 ) ) ; getchar ( ) ; return 0 ; }
#include <stdio.h> NEW_LINE int findMaximum ( int arr [ ] , int low , int high ) {
if ( low == high ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] >= arr [ high ] ) return arr [ low ] ;
if ( ( high == low + 1 ) && arr [ low ] < arr [ high ] ) return arr [ high ] ; int mid = ( low + high ) / 2 ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] > arr [ mid - 1 ] ) return arr [ mid ] ;
if ( arr [ mid ] > arr [ mid + 1 ] && arr [ mid ] < arr [ mid - 1 ] ) return findMaximum ( arr , low , mid - 1 ) ;
else return findMaximum ( arr , mid + 1 , high ) ; }
int main ( ) { int arr [ ] = { 1 , 3 , 50 , 10 , 9 , 7 , 6 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printf ( " The ▁ maximum ▁ element ▁ is ▁ % d " , findMaximum ( arr , 0 , n - 1 ) ) ; getchar ( ) ; return 0 ; }
int getMissingNo ( int a [ ] , int n ) { int i , total ; total = ( n + 1 ) * ( n + 2 ) / 2 ; for ( i = 0 ; i < n ; i ++ ) total -= a [ i ] ; return total ; }
int main ( ) { int a [ ] = { 1 , 2 , 4 , 5 , 6 } ; int miss = getMissingNo ( a , 5 ) ; printf ( " % d " , miss ) ; getchar ( ) ; }
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE void printTwoElements ( int arr [ ] , int size ) { int i ; printf ( " The repeating element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ abs ( arr [ i ] ) - 1 ] > 0 ) arr [ abs ( arr [ i ] ) - 1 ] = - arr [ abs ( arr [ i ] ) - 1 ] ; else printf ( " ▁ % d ▁ " , abs ( arr [ i ] ) ) ; } printf ( " and the missing element is " for ( i = 0 ; i < size ; i ++ ) { if ( arr [ i ] > 0 ) printf ( " % d " , i + 1 ) ; } }
int main ( ) { int arr [ ] = { 7 , 3 , 4 , 5 , 5 , 6 , 2 } ; int n = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoElements ( arr , n ) ; return 0 ; }
void printTwoOdd ( int arr [ ] , int size ) { int xor2 = arr [ 0 ] ;
int set_bit_no ;
int i ; int n = size - 2 ; int x = 0 , y = 0 ;
for ( i = 1 ; i < size ; i ++ ) xor2 = xor2 ^ arr [ i ] ;
set_bit_no = xor2 & ~ ( xor2 - 1 ) ;
for ( i = 0 ; i < size ; i ++ ) {
if ( arr [ i ] & set_bit_no ) x = x ^ arr [ i ] ;
else y = y ^ arr [ i ] ; } printf ( " The two ODD elements are % d & % d " }
int main ( ) { int arr [ ] = { 4 , 2 , 4 , 5 , 2 , 3 , 3 , 1 } ; int arr_size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; printTwoOdd ( arr , arr_size ) ; getchar ( ) ; return 0 ; }
bool findPair ( int arr [ ] , int size , int n ) {
int i = 0 ; int j = 1 ;
while ( i < size && j < size ) { if ( i != j && arr [ j ] - arr [ i ] == n ) { printf ( " Pair ▁ Found : ▁ ( % d , ▁ % d ) " , arr [ i ] , arr [ j ] ) ; return true ; } else if ( arr [ j ] - arr [ i ] < n ) j ++ ; else i ++ ; } printf ( " No ▁ such ▁ pair " ) ; return false ; }
int main ( ) { int arr [ ] = { 1 , 8 , 30 , 40 , 100 } ; int size = sizeof ( arr ) / sizeof ( arr [ 0 ] ) ; int n = 60 ; findPair ( arr , size , n ) ; return 0 ; }
void findFourElements ( int A [ ] , int n , int X ) {
for ( int i = 0 ; i < n - 3 ; i ++ ) {
for ( int j = i + 1 ; j < n - 2 ; j ++ ) {
for ( int k = j + 1 ; k < n - 1 ; k ++ ) {
for ( int l = k + 1 ; l < n ; l ++ ) if ( A [ i ] + A [ j ] + A [ k ] + A [ l ] == X ) printf ( " % d , ▁ % d , ▁ % d , ▁ % d " , A [ i ] , A [ j ] , A [ k ] , A [ l ] ) ; } } } }
int main ( ) { int A [ ] = { 10 , 20 , 30 , 40 , 1 , 2 } ; int n = sizeof ( A ) / sizeof ( A [ 0 ] ) ; int X = 91 ; findFourElements ( A , n , X ) ; return 0 ; }
const int cola = 2 , rowa = 3 , colb = 3 , rowb = 2 ;
void Kroneckerproduct ( int A [ ] [ cola ] , int B [ ] [ colb ] ) { int C [ rowa * rowb ] [ cola * colb ] ;
for ( int i = 0 ; i < rowa ; i ++ ) {
for ( int k = 0 ; k < rowb ; k ++ ) {
for ( int j = 0 ; j < cola ; j ++ ) {
for ( int l = 0 ; l < colb ; l ++ ) {
C [ i + l + 1 ] [ j + k + 1 ] = A [ i ] [ j ] * B [ k ] [ l ] ; printf ( " % d TABSYMBOL " , C [ i + l + 1 ] [ j + k + 1 ] ) ; } } printf ( " STRNEWLINE " ) ; } } }
int main ( ) { int A [ 3 ] [ 2 ] = { { 1 , 2 } , { 3 , 4 } , { 1 , 0 } } , B [ 2 ] [ 3 ] = { { 0 , 5 , 2 } , { 6 , 7 , 3 } } ; Kroneckerproduct ( A , B ) ; return 0 ; }
#include <stdio.h> NEW_LINE int Identity ( int num ) { int row , col ; for ( row = 0 ; row < num ; row ++ ) { for ( col = 0 ; col < num ; col ++ ) {
if ( row == col ) printf ( " % d ▁ " , 1 ) ; else printf ( " % d ▁ " , 0 ) ; } printf ( " STRNEWLINE " ) ; } return 0 ; }
int main ( ) { int size = 5 ; identity ( size ) ; return 0 ; }
void subtract ( int A [ ] [ N ] , int B [ ] [ N ] , int C [ ] [ N ] ) { int i , j ; for ( i = 0 ; i < N ; i ++ ) for ( j = 0 ; j < N ; j ++ ) C [ i ] [ j ] = A [ i ] [ j ] - B [ i ] [ j ] ; }
int main ( ) { int A [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int B [ N ] [ N ] = { { 1 , 1 , 1 , 1 } , { 2 , 2 , 2 , 2 } , { 3 , 3 , 3 , 3 } , { 4 , 4 , 4 , 4 } } ; int C [ N ] [ N ] ; int i , j ; subtract ( A , B , C ) ; printf ( " Result ▁ matrix ▁ is ▁ STRNEWLINE " ) ; for ( i = 0 ; i < N ; i ++ ) { for ( j = 0 ; j < N ; j ++ ) printf ( " % d ▁ " , C [ i ] [ j ] ) ; printf ( " STRNEWLINE " ) ; } return 0 ; }
#include <stdio.h>
#include <limits.h> NEW_LINE #include <stdio.h>
#include <stdio.h>
#include <limits.h> NEW_LINE #include <stdio.h>
#include <stdio.h>
#include <stdio.h>
int main ( ) {
#include <stdio.h>
#include <stdio.h> NEW_LINE #include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE #define bool  int NEW_LINE int findCandidate ( int * , int ) ; bool isMajority ( int * , int , int ) ;
#include <stdio.h> NEW_LINE #include <stdlib.h>
#include <stdio.h> NEW_LINE #include <stdlib.h> NEW_LINE #include <math.h>
#include <stdio.h> NEW_LINE #include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
int max ( int a , int b ) { return ( a > b ) ? a : b ; }
#include <stdio.h> NEW_LINE #include <string.h>
int max ( int x , int y ) { return ( x > y ) ? x : y ; }
#include <stdbool.h> NEW_LINE #include <stdio.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE #include <string.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE #include <string.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE void combinationUtil ( int arr [ ] , int n , int r , int index , int data [ ] , int i ) ;
#include <stdio.h> NEW_LINE #include <stdlib.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE #define bool  int
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
# include <stdio.h> NEW_LINE # define bool  int
#include <stdio.h> NEW_LINE #define bool  int
#include <stdio.h>
int main ( ) {
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h> NEW_LINE #define N  4
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
