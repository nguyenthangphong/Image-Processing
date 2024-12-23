#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

void convert_jpeg_to_rgb(const char *input_filename, const char *output_filename)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    /* Step 1: Allocate and initialize JPEG decompression object */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    /* Step 2: Specify data source */
    FILE *infile = fopen(input_filename, "rb");
    
    if (infile == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", input_filename);
        exit(1);
    }
    jpeg_stdio_src(&cinfo, infile);

    /* Step 3: Read the JPEG header */
    jpeg_read_header(&cinfo, TRUE);

    /* Step 4: Start decompressor */
    jpeg_start_decompress(&cinfo);

    /* Step 5: Allocate memory for the RGB image */
    int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *rgb_buffer = (unsigned char *)malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

    if (rgb_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for RGB buffer\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        exit(1);
    }

    /* Step 6: Read the scanlines */
    unsigned char *row_pointer[1];

    while (cinfo.output_scanline < cinfo.output_height)
    {
        row_pointer[0] = rgb_buffer + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    /* Step 7: Finish decompression */
    jpeg_finish_decompress(&cinfo);

    /* Step 8: Save the RGB buffer to a text file */
    FILE *outfile = fopen(output_filename, "w");

    if (outfile == NULL)
    {
        fprintf(stderr, "Cannot open %s for writing\n", output_filename);
        free(rgb_buffer);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        exit(1);
    }

    for (int y = 0; y < cinfo.output_height; y++)
    {
        for (int x = 0; x < cinfo.output_width; x++)
        {
            int index = (y * cinfo.output_width + x) * cinfo.output_components;
            fprintf(outfile, "%d %d %d\n", rgb_buffer[index], rgb_buffer[index + 1], rgb_buffer[index + 2]);
        }
    }

    fclose(outfile);

    /* Step 9: Clean up */
    free(rgb_buffer);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    convert_jpeg_to_rgb(argv[1], argv[2]);

    printf("JPEG converted to RGB and saved to %s\n", argv[2]);
    return 0;
}
