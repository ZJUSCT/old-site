---
title: CUDA内存管理总结(一)
date: 2018-11-25 00:46:08
author: 潘薇鸿
tags: [Tech, CUDA]
---

# CUDA 内存管理(一)

## 一、寄存器

​	GPU的每个SM（流多处理器）都有上千个寄存器，每个SM都可以看作是一个多线程的CPU核，但与一般的CPU拥有二、四、六或八个核不同，一个GPU可以有**N个SM核**；同样，与一般的CPU核支持一到两个硬件线程不同，每个SM核可能有**8~192个SP**（流处理器），亦即每个SM能同时支持这么多个硬件线程。事实上，一台GPU设备的所有SM中活跃的线程数目通常数以万计。

### 1.1 寄存器映射方式

​	**CPU处理多线程**：进行上下文切换，使用寄存器重命名机制，将当前所有寄存器的状态保存到栈（系统内存），再从栈中恢复当前需要执行的新线程在上一次的执行状态。这些操作通常花费上百个CPU时钟周期，有效工作吞吐量低。

​	**GPU处理多线程**：与CPU相反，GPU利用多线程隐藏了内存获取与指令执行带来的延迟；此外，GPU不再使用寄存器重命名机制，而是尽可能为每个线程分配寄存器，从而上下文切换就变成了寄存器组选择器（或指针）的更新，几乎是零开销。

### 1.2 寄存器空间大小

​	每个SM可提供的寄存器空间大小分别有8KB、16KB、32KB和64KB，每个线程中的每个变量占用一个寄存器，因而总共会占用N个寄存器，N代表调度的线程数量。当线程块上的寄存器数目是允许的最大值时，每个SM会只处理一个线程块。

### 1.3 SM调度线程、线程块

​	由于大多数内核对寄存器的需求量很低，所以可以通过降低寄存器的需求量来增加SM上线程块的调度数量，从而提高运行的线程总数，根据线程级并行**“占用率越高，程序运行越快”**，可以实现运行效率的优化。当线程级并行（*Thread-Level Parallelism*，TLP）足以隐藏存储延迟时会达到一个临界点，此后想要继续提高程序性能，可以在单个线程中实现指令级的并行（*Instruction-Level Parallelism*，ILP），即单线程处理多数据。

​	但在另一方面，每个SM所能调度的线程总量是有限制的，因此当线程总量达到最大时，再减少寄存器的使用量就无法达到提高占有率的目的（如下表中寄存器数目由20减小为16，线程块调度数量不变），所以在这种情况下，应增加寄存器的使用量到临界值。

<img src="1.png" />

### 1.4 寄存器优化方式

​	1）将中间结果累积在寄存器而非全局内存中。尽量避免全局内存的写操作，因为如果操作聚集到同一块内存上，就会强制硬件对内存的操作序列化，导致严重的性能降低；

​	2）循环展开。循环一般非常低效，因为它们会产生分支，造成流水线停滞。

### 1.5 总结

​	使用寄存器可以有效消除内存访问，或提供额外的ILP，以此实现GPU内核函数的加速，这是最为有效的方法之一。

## 二、共享内存

### 2.1 基本概念

​	1、共享内存实际上是可以受用户控制的一级缓存，每个SM中的一级缓存和共享内存共用一个64KB的内存段。

​	2、共享内存的延迟很低，大约有1.5TB/s的带宽，而全局内存仅为160GB/s，换言之，有效利用共享内存有可能获得7倍的加速比。但它的速度依然只有寄存器的十分之一，并且共享内存的速度几乎在所有GPU中都相同，因为它由核时钟频率驱动。

​	3、只有当数据重复利用、全局内存合并，或者线程之间有共享数据（例如同时访问相同地址的存储体）的时候使用共享内存才更合适，否则将数据直接从全局内存加载到寄存器性能会更好。

​	4、共享内存是基于存储体切换的架构（*bank-switched architecture*），费米架构的设备上有32个存储体。无论有多少线程发起操作，**每个**存储体**每个**周期只执行**一次**操作。因此，如果线程束中的每个线程各访问一个存储体，那么所有线程的操作都可以在一个周期内同时执行，且所有操作都是独立互不影响的。此外，如果所有线程同时访问同一地址的存储体，会触发一个广播机制到线程束中的每个线程中。但是，如果是其他的访问方式，线程访问共享内存就需要排队，即一个线程访问时，其他线程将阻塞闲置。因此很重要的一点时，应该尽可能地获得**零存储体冲突**的共享内存访问。

### 2.2 Example：使用共享内存排序

## 2.2.1 归并排序

​	假设待排序的数据集大小为N，现将数据集进行划分。根据归并排序的划分原则，最后每个数据包中只有两个数值需要排序，因此，在这一阶段，最大并行度可达到 $N \over 2$ 个独立线程。例如，处理一个大小为512KB的数据集，共有128K个32位的元素，那么最多可以使用的线程个数为64K个（N=128K，N/2=64K），假设GPU上有16个SM，每个SM最多支持1536个线程，那么每个GPU上最多可以支持24K个线程，因此，按照这样划分，64K的数据对只需要2.5次迭代即可完成排序操作。

​	但是，如果采用上述划分排序方式再进行合并，我们需要从每个排好序的数据集中读出元素，对于一个64K的集合，需要64K次读操作，即从内存中获取256MB的数据，显然当数据集很大的时候不合适。

​	因此，我们采用通过限制对原始问题的迭代次数，通过基于共享内存的分解方式来获得更好的合并方案。因为在费米架构的设备上有32个存储体，即对应32个线程，所以当需要的线程数量减少为32（一个线程束）时，停止迭代，于是共需要线程束4K个（128K/32=4K），又因为GPU上有16个SM，所以这将为每个SM分配到256个线程束。然而由于费米架构设备上的每个SM最多只能同时执行48个线程束，因此多个块将被循环访问。

​	通过将数据集以每行32个元素的方式在共享内存中进行分布，每列为一个存储体，即可得到零存储体冲突的内存访问，然后对每一列实施相同的排序算法。（或者也可以理解为桶排序呀）

​	然后再进行列表的合并。

## 2.2.2 合并列表

​	先从串行合并任意数目的有序列表看起：

```c
void merge_array(const u32 *const src_array, //待排序数组
                 u32 *const dest_array, //排序后的数组
                 const u32 num_lists, //列表总数
                 const u32 num_elements) //数据总数
{
	const u32 num_elements_per_list = (num_elements / num_lists);//每个列表中的数据个数
    u32 list_indexes[MAX_NUM_LISTS]; //所有列表当前所在的元素下标
    for(u32 list = 0; list < num_lists; list++)
    {
		list_indexes[list] = 0;
    }
    for(u32 i = 0; i<num_elements; i++)
    {
		dest_array[i] = find_min(scr_array, 
                                 list_indexes, 
                                 num_lists, 
                                 num_elements_per_list);
	}
}

u32 find_min(const u32*cosnt src_array, 
             u32 *const list_indexes, 
             const u32 num_lists, 
             const u32 num_elements_per_list)//寻找num_lists个元素中的最小值
{
    u32 min_val = 0xFFFFFFFF;
    u32 min_idx = 0;
    
    for(u32 i = 0; i < num_lists; i++)
    {
		if(list_indexes[i] < num_elements_per_list)
        {
			const u32 src_idx = i + (list_indexes[i]*num_lists);
             const u32 data = src_array[src_idx];
        
        	if(data <= min_val)
        	{
				min_val = data;
           	     min_idx = i;
        	}
        }
    }
    
    list_indexes[min_idx]++;
    return min_val;
}
```

​	将上述算法用GPU实现

```C
__global__ void gpu_sort_array_array(u32 *const data, 
                                     const u32 num_lists, 
                                     const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ u32 sort_tmp[NUM_ELEM];
    __shared__ u32 sort_tmp_1[NUM_ELEM];
    
    copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);
    radix_sort2(sort_tmp, num_lists, num_elements, tid, sort_tmp_1);
    merge_array6(sort_tmp, data, num_lists, num_elements, tid);
}
```

​	第一个函数的实现：

```C
__device__ void copy_data_to_shared(const u32 *const data, 
                                    u32 *sort_tmp, 
                                    const u32 num_lists, 
                                    const u32 num_elements, 
                                    const u32 tid)
{
    for(u32 i = 0; i < num_elements; i++)
    {
		sort_tmp[i+tid] = data[i+tid]; 
    }
    __syncthreads();
}
```

​	该函数中，程序按行将数据从全局内存读入共享内存。当函数调用一个子函数并传入参数时，这些参数必须以某种方式提供给被调用的函数，有两种方法可以采用。一种是通过寄存器传递所需的值，另一种方法是创建一个名为“栈帧”的内存区，但这种方法非常地不高效。出于这一原因，我们需要重新修改合并的程序(merge_array)，以避免函数调用，修改后程序如下（单线程）：

```C 
__device__ void merge_array1(const u32 *const src_array, 
                             u32 *const dest_array, 
                             const u32 num_lists, 
                             const u32 num_elements, 
                             const u32 tid)
{
	__shared__ u32 list_indexes[MAX_NUM_LISTS];
    
    lists_indexes[tid] = 0;//从每个列表的第一个元素开始
    __syncthreads();
    
    //单线程
    if(tid == 0)
    {
		const u32 num_elements_per_list = (num_elements / num_lists);
        for(u32 i = 0; i < num_elements; i++)
        {
			u32 min_val = 0xFFFFFFFF;
             u32 min_idx = 0;
            
            for(u32 list = 0; list < num_lists; list++)
   		   {
			if(list_indexes[list] < num_elements_per_list)
        	{
			   const u32 src_idx = i + (list_indexes[i]*num_lists);
             	const u32 data = src_array[src_idx];
        		if(data <= min_val)
        		{
					min_val = data;
           	     	 min_idx = i;
                }
        	}
          }
          list_indexes[min_idx]++;
          dest_array[i]=min_val;
       }
    }
}
```

​	这里只用一个线程进行合并，但显然，为了获得更好的性能，一个线程是远远不够的。因为数据被写到一个单一的列表中，所以多个线程必须进行某种形式的合作。

```C
__device__ void merge_array6(const u32 *const src_array, 
                             u32 *const dest_array, 
                             const u32 num_lists, 
                             const u32 num_elements, 
                             const u32 tid)
{
    //每个列表分到的元素个数
	const u32 num_elements_per_list = (num_elements / num_lists);
    
    //创建一个共享列表数组，用来储存当前线程所访问的列表元素下标
    __shared__ u32 list_indexes[MAX_NUM_LISTS];
    list_indexes[tid] = 0;
    
    //创建所有线程共享的最小值与最小值线程号
    __shared__ u32 min_val;
    __shared__ u32 min_tid;
    __syncthreads();
    
    for(u32 i=0; i<num_elements; i++)
    {   
        u32 data;
        //如果当前列表还未被读完，则从中读取数据
        if(list_indexes[tid] < num_elements_per_list);
        {
             //计算出当前元素在原数组中的下标
			const u32 src_idx = tid + (list_indexes[tid] * num_lists);
             data = src_array[src_idx];
        }
        else
        {
			data = 0xFFFFFFFF;
        }
        
        //用零号线程来初始化最小值与最小值线程号
        if(tid == 0)
        {
			min_val = 0xFFFFFFFF;
             min_tid = 0xFFFFFFFF;
        }
        __syncthreads();
        
        //让所有线程都尝试将它们现在手上有的值写入min_val，但只有最小的数据会被保留
        //利用__syncthreads()确保每个线程都执行了该操作
        atomicMin(&min_val, data);
        __syncthreads();
        
        //在所有data==min_val的线程中，选取最小线程号写入min_tid
        if(min_val == data)
        {
			atomicMin(&min_tid, tid);
        }
        __syncthreads();
        
        //将满足要求的线程所在列表的当前元素往后移一位，进行下一轮比较
        //并将筛选结果存入结果数组dest_array
        if(tid == min_tid)
        {
			list_indexes[tid]++;
             dest_array[i] = data;
        }
    }
}
```

​	上面的函数中将num_lists个线程进行合并操作，但只用了一个线程一次将结果写入结果数据数组中，保证了结果的正确性，不会引起线程间的冲突。

​	其中使用到了 atomicMin 函数。每个线程以从列表中获取的数据作为入参调用该函数，取代了原先单线程访问列表中所有元素并找出最小值的操作。当每个线程调用 atomicMin 函数时，线程读取保存在共享内存中的最小值并于当前线程中的值进行比较，然后把比较结果重新写回最小值对应的共享内存中，同时更新最小值对应的线程号。然而，由于列表中的数据可能会重复，因此可能出现多个线程的值均为最小值的情况，保留的线程号却各不相同。因此需要执行第二步操作，保证保留的线程号为最小线程号。

​	虽然这种方法的优化效果很显著，但它也有一定的劣势。例如，atomicMin函数只能用在计算能力为1.2以上的设备上；另外，aotomicMin函数只支持整数型运算，但现实世界中的问题通常是基于浮点运算的，因此在这种情况下，我们需要寻找新的解决方法。

## 2.2.3 并行归约

​	并行归约适用于许多问题，求最小值只是其中的一种。它使用数据集元素数量一半的线程，每个线程将当前线程对应的元素与另一个元素进行比较，计算两者之间的最小值，并将得到的最小值移到前面。每进行一次比较，线程数减少一半，如此反复直到只剩一个元素为止，这个元素就是需要的最小值。

​	在选择比较元素的时候，应该尽量避免选择同一个线程束中的元素进行比较，因为这会明显地导致线程束内产生分支，而每个分支都将使SM做双倍的工作，继而影响程序的性能。因此我们选择将线程束中的元素与另一半数据集中的元素进行比较。如下图，阴影部分表示当前活跃的线程。

<img src="2.png" />

```C
__device__ void merge_array5(const u32 *const src_array, 
                             u32 *const dest_array, 
                             const u32 num_lists,
                             const u32 num_elements, 
                             const u32 tid)
{
	const u32 num_elements_per_list = (num_elements / num_lists);
    
    __shared__ u32 list_indexes[MAX_NUM_LISTS];
    __shared__ u32 reduction_val[MAX_NUM_LISTS];
    __shared__ u32 reduction_idx[MAX_NUM_LISTS];
    
    list_indexes[tid] = 0;
    reduction_val[tid] = 0;
    reduction_idx[tid] = 0;
    __syncthreads();
    
    for(u32 i=0; i<num_elements; i++)
    {
		u32 tid_max = num_lists >> 1;//最大线程数为列表总数的一半
         u32 data;//使用寄存器可以提高运行效率，将对共享内存的写操作次数减少为1
        
        //当列表中还有未处理完的元素时
         if(list_indexes[tid] < num_elements_per_list)
         {
             //计算该元素在原数组中的位置
			cosnst u32 src_idx = tid + (list_indexes[tid] * num_lists);
             data = src_array[src_idx];
         }
        //若当前列表已经处理完，将data赋值最大
        else
        {
			data = 0xFFFFFFFF;
        }
        
        //将当前元素及线程号写入共享内存
        reduction_val[tid] = data;
        reduction_idx[tid] = tid;
        __syncthreads;
        
        //当前活跃的线程数多于一个时
        while(tid_max!=0)
        {
            if(tid < tid_max)
            {
                 //将当前线程中的元素与另一半数据集中的对应元素进行比较
				const u32 val2_idx = tid + tid_max;
                 const u32 val2 = reduction_val[val2_idx];
                 
                 //最后保留较小的那个元素
                 if(reduction_val[tid] > val2)
                 {
					reduction_val[tid] = val2;
                      reduction_idx[tid] = reduction_idx[val_idx];
                 }
            }
            
            //线程数减半，进入下一轮循环
            tid_max >>= 1;
            __syncthreads();
        }
        
        //在零号线程中将结果写入结果数组，并将相应线程所指的元素后移一位
        if(tid == 0)
        {
			list_indexes[reduction_idx[0]]++;
             dest_array[i] = reduction_val[0];
        }
        __syncthreads();
    }
}
```

​	同样，这种方法也在共享内存中创建了一个临时的列表 list_indexes 用来保存每次循环中从 num_list 个数据集列表中选取出来进行比较的数据。如果进行合并的列表已经为空，那么就将临时列表中的对应数据区赋最大值0xFFFFFFFF。而每轮while循环后，活跃的线程数都将减少一半，直到最后只剩一个活跃的线程，亦即零号线程。最后将结果复制到结果数组中并将最小值所对应的列表索引加一，以确保元素不会被处理两次。

## 2.2.4 混合算法

​	在了解atomicMin函数和并行归约两种方案后，我们可以利用这两种算法各自的优点，创造出一种新的混合方案。

​	简单的1~N个数据归约的一个主要问题就是当N增大时，程序的速度先变快再变慢，达到最高效的情形时N在8至16左右。混合算法将原数据集划分成诸多个小的数据集，分别寻找每块中的最小值，然后再将每块得到的结果最终归约到一个值中。这种方法和并行归约的思想非常相似，但同时又省略了并行归约中的多次迭代。代码更新如下：

```C
#define REDUCTION_SIZE 8
#define REDUCTION_SIZE_BIT_SHIFT 3
#define MAX_ACTIVE_REDUCTIONS ((MAX_NUM_LISTS) / (REDUCTION_SIZE))

__device__ void merge_array(const u32 *const src_array, 
                            u32 *const dest_array, 
                            const u32 num_lists, 
                            const u32 num_elements, 
                            const u32 tid)
{
    //每个线程都从原数组中读入一个数据，用作首次比较
    u32 data = src_array[tid];
    
    //当前线程所在的数据块编号（8个线程为一组，每个线程处理一个列表）
    const u32 s_idx = tid >> REDUCTION_SIZE_BIT_SHIFT;
    
    //首次进行分别归约的数据块总数
    const u32 num_reductions = num_lists >> REDUCTION_SIZE_BIT_SHIFT;
    const u32 num_elements_per_list = num_elements / num_lists;
    
    //在共享内存中创建一个列表，指向每个线程当前所在的元素，并初始化为0
    __shared__ u32 list_indexes[MAX_NUM_LISTS];
    list_indexes[tid] = 0;
    
    //遍历所有数据
    for(u32 i=0; i<num_elements; i++)
    {
        //每个数据块在内部归约后都会产生一个相应的最小值
        //在共享内存中开辟一个列表，用来保存每组的最小值
		__shared__ u32 min_val[MAX_ACTIVE_REDUCTIONS];
         __shared__ u32 min_tid;
        
        //初始化每个数据块的内部最小值
        if(tid < num_lists)
        {
			min_val[s_idx] = 0xFFFFFFFF;
             min_tid = 0xFFFFFFFF;
        }
        __syncthreads();
        
        //将当前线程的数据与所处数据块的最小值进行比较，并保留较小的那一个
        atomicMin(&min_val[s_idx], data);
        
        //进行归约的数据块总数不为零时
        if(num_reductions > 0)
        {
            //确保每个线程都已经将上一步比较操作完成
			__syncthreads();
            
             //将每个数据块产生的最小值与零号数据块的最小值进行比较，保留较小的那一个
             if(tid < num_reductions)
             {
				atomicMin(&min_val[0], min_val[tid]);
                  __syncthreads();
             }
            
             //如果当前线程的数据等于此次比较保留的最小值，记录最小线程号
             if(data == min_val[0])
             {
				atomicMin(&min_tid, tid);
             }
             //确保上一步操作每个线程都已经完成，才能执行下一句
             __syncthreads();
            
            //如果当前线程号恰为记录下的最小线程号
            if(tid == min_tid)
            {
                 //当前所指元素后移一位
				list_indexes[tid]++;
                
                 //将结果保存入结果数组
                  dest_array[i] = data;
                 
                  //若该线程对应的列表尚未被处理完
                  if(list_indexes[tid] < num_elements_per_list)
                      //更新该线程的data，进行下一轮比较
                      data = src_array[tid + (list_indexes[tid] * num_lists)];
                  else
                      data = 0xFFFFFFFF;
            }
            __syncthreads();
        }
    }
}
```

​	注意到：

​	1）原来的min_val由单一的数据扩展成为一个共享数据的数组，这是因为每个独立的线程都需要从它对应的数据集中获取当前的最小值来进行内部比较。每个最小值都是一个32位的数值，因此可以存储在独立的共享内存存储体中。

​	2）内核函数中的REDUCTION_SIZE的值被设置成8，意味着每个数据块中包含8个数据，程序分别找出每个数据块的最小值，然后再在这些最小值中寻找最终的最小值。

​	3）内核函数中最重要的一个变化是，只有每次比较的最小值所对应的那个线程的data才会更新，其他线程的data都不会更新。而在之前的内核函数中，每轮比较开始，所有线程都会从对应的列表中重新读入data 的值，随着N的增大，这将变得越来越低效。

## 2.2.5 总结

​	1）共享内存允许同一个线程块中的线程读写同一段内存，但线程看不到也无法修改其他线程块的共享内存。

​	2）共享内存的缓冲区驻留在物理GPU上，所以访问时的延迟远低于访问普通缓冲区的延迟，因此除了使用寄存器，还应更有效地使用共享内存，尤其当数据有重复利用，或全局内存合并，或线程间有共享数据的时候。

​	3）编写代码时，将关键字_shared__添加到声明中，使得该变量留驻在共享内存中，并且线程块中的每个线程都可以共享这块内存，使得一个线程块中的多个线程能够在计算上进行通信和协作。

​	4）调用 __syncthreads() 函数来实现线程的同步操作，尤其要注意确保在读取共享内存之前，想要写入的操作都已经完成。另外还需要注意，切不可将这个函数放置在发散分支（某些线程需要执行，而其他线程不需要执行），因为除非线程块中的每个线程都执行了该函数，没有任何线程能够执行之后的指令，从而导致死锁。

​	5）不妨尝试使用共享内存实现矩阵乘法的优化。

 > Author: 潘薇鸿
 > PostDate: 2018.11.25

