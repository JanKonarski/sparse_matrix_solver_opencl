typedef int int32_t;

__kernel void solver(int32_t dim,
					 int32_t number,
					 __global int32_t* rows,
					 __global int32_t* cols,
					 __global float* A,
					 __global float* b,
					 __global float* x,
					 __local float* apa,
					 __local float* pa,
					 __local float* g)
{
	size_t id = get_local_id(0);

	x[id] = 0;
	pa[id] = g[id] = b[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	int32_t start, stop;
	start = stop = -1;

	for (size_t i=id; i<number; ++i)
	{
		if (rows[i] == id && start == -1)
			start = i;

		else if (i == number-1 && stop == -1)
			stop = i;

		else if (rows[i] == id+1 && stop == -1)  {
			stop = i-1;
			break;
		}
	}

	local float lr;
	local float spr;

	if (id == 0)
	{
		spr = 0;
		for (size_t i=0; i<dim; i++)
			spr += pow(g[i], 2);
		lr = sqrt(spr);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float app;
	size_t it = 0;
	while (lr >= 1e-45 &&
		   it < 1000000)
	{
		if (id == 0)
			it++;

		apa[id] = 0;
		for(size_t i=start; i<=stop; ++i)
			apa[id] += pa[cols[i]] * A[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		local float alpha;
		if (id == 0)
		{
			app = 0;
			for(size_t i=0; i<dim; ++i)
				app += apa[i] * pa[i];
			alpha = spr / app;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		g[id] -= apa[id] * alpha;
		x[id] += pa[id] * alpha;
		barrier(CLK_LOCAL_MEM_FENCE);

		local float pr;
		if (id == 0)
		{
			pr = 0;
			for (size_t i=0; i<dim; ++i)
				pr += pow(g[i], 2);
			lr = sqrt(pr);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		pa[id] = g[id] + pa[id] * (pr / spr);
		barrier(CLK_LOCAL_MEM_FENCE);

		spr = pr;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
