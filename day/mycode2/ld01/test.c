
#include <stdio.h>
 
#ifdef __cplusplus
extern "C"{
    
#endif
    
void add(int num)
{
    printf("*****************************************\n");
    printf("This is ok, the number is okay. %d\n", num);
    printf("*****************************************\n");
}
 
 
#ifdef __cplusplus
    }
    
#endif

