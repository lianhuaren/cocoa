
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dlfcn.h> /* 必须加这个头文件 */
#include <assert.h>
 
int main(int argc, char *argv[])
{
    printf("come in\n");
 
    void *handler = dlopen("libtest.so", RTLD_NOW);
    assert(handler != NULL);
    
    void (*pTest)(int);
    pTest = (void (*)(int))dlsym(handler, "add");
    
    (*pTest)(10);
 
    dlclose(handler);
    
    printf("go out\n");
    return 0;
}

