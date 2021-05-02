#ifndef MAKE_GRID
#define MAKE_GRID

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef double data_t;

/**************************************************************
						AGENT CLASS
**************************************************************/

struct agent{
	int sx;
	int sy;
	int dx;
	int dy;
	char* name;
	struct agent* next;
};

/**************************************************************
						GRID CLASS
**************************************************************/

typedef struct{
	int num_agents;
	struct agent* agents;
	int size;
	int dx;
	int dy;
	data_t* diff_matrix;
	data_t* obs_matrix;
} grid;

grid* LoadGrid(char* filename){
	// Reads from YAML file to initialize grid

	// Variable declarations
	int i, j, mult, size, mod_size, mod_size2;
	int x, y, sx, sy, dx, dy;
	FILE *fp;
	char* line = NULL;
	char* token;
	char* name;
	size_t line_size = 0;
	struct agent* next_agent;

	// Allocate grid
	grid* example = (grid*) calloc(1, sizeof(grid));
	example->num_agents = 0;

	// Open file
	fp = fopen(filename, "r");
	if (!fp){
		printf("Cannot open file\n");
		exit(1);
	}

	// Line Parser/load data
	if (!getline(&line, &line_size, fp)){ // agents
		printf("Invalid file\n");
		exit(1);
	}

	// handle agent info
	while (getline(&line, &line_size, fp)){
		if (strncmp(line, "map:", 4) == 0){
			break;
		}

		// read in goal
		token = strtok(line, "["); // remove padding
		token = strtok(NULL, ", "); // first digit
		dx = atoi(token) + 1;
		token = strtok(NULL, "]"); // second digit
		token = strcpy(token, token+1);
		dy = atoi(token) + 1;

		// read in name
		if (!getline(&line, &line_size, fp)){
			printf("Invalid file\n");
			exit(1);
		}
		token = strtok(line, ":"); // remove padding
		token = strtok(NULL, " "); // remove padding
		name = strtok(NULL, " "); // read name

		// read in start
		if (!getline(&line, &line_size, fp)){
			printf("Invalid file\n");
			exit(1);
		}
		token = strtok(line, "["); // remove padding
		token = strtok(NULL, ", "); // first digit
		sx = atoi(token) + 1;
		token = strtok(NULL, "]"); // second digit
		token = strcpy(token, token+1);
		sy = atoi(token) + 1;

		if (!example->num_agents){
			example->agents = (struct agent*) calloc(1, sizeof(struct agent));
			example->dx = dx;
			example->dy = dy;
			example->agents->dx = dx;
			example->agents->dy = dy;
			example->agents->sx = sx;
			example->agents->sy = sy;
			example->agents->name = name;
			example->agents->next = NULL;
			next_agent = example->agents;
		}
		else{
			next_agent->next = (struct agent*) calloc(1, sizeof(struct agent));
			next_agent = next_agent->next;
			next_agent->dx = dx;
			next_agent->dy = dy;
			next_agent->sx = sx;
			next_agent->sy = sy;
			next_agent->name = name;
			next_agent->next = NULL;
		}

		// increment agent count
		example->num_agents++;
	}

	// get dimensions
	if (!getline(&line, &line_size, fp)){
		printf("Invalid file\n");
		exit(1);
	}
	token = strtok(line, "["); // remove padding
	token = strtok(NULL, ", "); // first digit
	size = atoi(token);
	mod_size = size + 1;
	mod_size2 = size + 2;
	example->size = size;
	token = strtok(NULL, "]"); // second digit
	token = strcpy(token, token+1);

	// initialize obstacle matrix w/ 1s
	example->obs_matrix = (data_t*) calloc((mod_size2) * (mod_size2), sizeof(data_t));
	for (i = 0; i < mod_size2; i++){
		mult = i*mod_size2;
		for (j = 0; j < mod_size2; j++){
			example->obs_matrix[mult + j] = 1;
		}
	}

	// change added edges of obs_matrix to 0
	for (i = 0; i < mod_size2; i += (mod_size2-1)){
		mult = i*mod_size2;
		for (j = 0; j < mod_size2; j++){
			example->obs_matrix[mult + j] = 0;
		}
	}
	for (i = 0; i < mod_size2; i += (mod_size2-1)){
		mult = i*mod_size2;
		for (j = 0; j < mod_size2; j++){
			example->obs_matrix[j*mod_size2 + i] = 0;
		}
	}

	// initialize diffusion matrix w/ 1s
	example->diff_matrix = (data_t*) calloc((mod_size2) * (mod_size2), sizeof(data_t));
	for (i = 0; i < mod_size2; i++){
		mult = i*mod_size2;
		for (j = 0; j < mod_size2; j++){
			example->diff_matrix[mult + j] = 0;
		}
	}
	example->diff_matrix[example->dx + example->dy * mod_size2] = (data_t) size * size;

	if (!getline(&line, &line_size, fp)){
		printf("Invalid file\n");
		exit(1);
	} // remove padding

	// Load obstacle info into matrix
	while (getline(&line, &line_size, fp) > 1){
		token = strtok(line, "["); // remove padding
		token = strtok(NULL, ", "); // first digit
		x = atoi(token) + 1;
		token = strtok(NULL, "]"); // second digit
		token = strcpy(token, token+1);
		y = atoi(token) + 1;

		example->obs_matrix[x + y*mod_size2] = 0; // obstacle to 0
	}

	fclose(fp);

	return example;
}

int PrintGrid(grid* example){
	int i, j, mult;
	//struct agent* next_agent = example->agents;
	int size = example->size + 2;

	for (i = 0; i < size; i++){
		mult = i*size;
		for (j = 0; j < size; j++){
			printf("%.0f,\t", example->diff_matrix[mult + j]);
		}
		printf("\n");
	}

	printf("\n");

	for (i = 0; i < size; i++){
		mult = i*size;
		for (j = 0; j < size; j++){
			printf("%.0f,\t", example->obs_matrix[mult + j]);
		}
		printf("\n");
	}
/*
	while (next_agent != NULL){
		printf("[%d, %d]\n", next_agent->sx, next_agent->sy);
		next_agent = next_agent->next;
	}
*/
	return 0;
}

#endif
