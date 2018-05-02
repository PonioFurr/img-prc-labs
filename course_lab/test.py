import course_lab.sign_replacer as sr
import os

path = os.path.dirname(__file__)

sr.process(os.path.join(path, 'test\\test_task.json'),
           path,
           os.path.join(path, 'russian_signs\\description\\'),
           os.path.join(path, 'russian_signs\\images\\'),
           os.path.join(path, 'test\\output'),
           True)
